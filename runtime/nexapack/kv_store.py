"""Private, bounded CPU backing files for complete cold Q3 KV pages.

Files are session scratch, not checkpoints or NexaPack weights. The caller owns
the source/destination memory and guarantees its lifetime and exclusive access
during each call. Reads can partially overwrite the supplied slot on failure;
the caller must discard that slot until verification succeeds. Python metadata,
file descriptors, hash state and OS cache are outside the raw-buffer bound.

A descriptor must come from the canonical tier plan. In particular its cold Q3
page is complete: TieredKVPage does not expose enough geometry to independently
reconstruct the full layout or prove completeness for logical page zero.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import uuid

from compiler.kv_plan import ALIGNMENT
from compiler.paged_kv_plan import MAX_ALLOCATION_BYTES
from compiler.tiered_kv_plan import TieredKVPage


KV_STORE_MAGIC = b"NEXAKV01"
KV_STORE_VERSION = 1
KV_STORE_HEADER = struct.Struct("<8sHHIQ32s")
KV_STORE_MAX_METADATA_BYTES = 8192
KV_STORE_CHUNK_BYTES = 65536
# At most two encoded metadata buffers, plus bounded header/digest/EOF staging.
# No owned payload buffer: payload views borrow the caller's page or reload slot.
KV_STORE_BUFFER_BYTES = 2 * KV_STORE_MAX_METADATA_BYTES + 512
_MAX_POINTER = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1


def _integer(value, name, minimum=1, maximum=MAX_ALLOCATION_BYTES):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _pointer(address, size):
    _integer(address, "page address", maximum=_MAX_POINTER)
    if address > _MAX_POINTER - size:
        raise ValueError("Page address range overflows the native pointer range")


def _validate_descriptor(descriptor, extent_bytes):
    if not isinstance(descriptor, TieredKVPage):
        raise ValueError("KV backing store requires a TieredKVPage descriptor")
    # Revalidate even if a frozen dataclass was constructed through a bypass.
    TieredKVPage.from_dict(descriptor.to_dict())
    _integer(extent_bytes, "extent_bytes")
    if (descriptor.codec != "q3" or descriptor.tier != "cold"
            or descriptor.logical_start != descriptor.page_index * descriptor.valid_tokens):
        raise ValueError("KV backing store accepts only complete cold Q3 pages")
    if (extent_bytes % ALIGNMENT or descriptor.page_allocation_bytes != extent_bytes + ALIGNMENT - 1):
        raise ValueError("KV extent must exclude exactly the page's alignment slack")
    for name in ("page_index", "logical_start", "age_rank", "valid_tokens"):
        _integer(getattr(descriptor, name), name, minimum=0 if name != "valid_tokens" else 1)


def _metadata(identity, descriptor, extent_bytes, checksum):
    value = {"format": "NexaKVPage", "version": KV_STORE_VERSION,
             "identity": identity, "descriptor": descriptor.to_dict(),
             "extent_bytes": extent_bytes, "payload_sha256": checksum}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    if len(encoded) > KV_STORE_MAX_METADATA_BYTES:
        raise ValueError("KV backing metadata exceeds the 8192-byte limit")
    return encoded


def _header(metadata, extent_bytes):
    return KV_STORE_HEADER.pack(KV_STORE_MAGIC, KV_STORE_VERSION, 0, len(metadata),
                                extent_bytes, hashlib.sha256(metadata).digest())


def _open_file(directory, name, flags):
    options = {"dir_fd": directory} if isinstance(directory, int) else {}
    path = name if isinstance(directory, int) else directory / name
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                 0o600, **options)
    try:
        return io.FileIO(fd, "r" if flags & os.O_ACCMODE == os.O_RDONLY else "w", closefd=True)
    except BaseException:
        os.close(fd)
        if flags & os.O_EXCL and flags & os.O_CREAT:
            os.unlink(path, **options)
        raise


def _write_all(stream, data, digest=None):
    view = chunk = written = None
    try:
        view = memoryview(data).cast("B")
        offset = 0
        while offset < len(view):
            chunk = view[offset:offset + KV_STORE_CHUNK_BYTES]
            try:
                count = stream.write(chunk)
                if type(count) is not int or not 0 < count <= len(chunk):
                    raise OSError("KV backing write made no progress or exceeded its buffer")
                if digest is not None:
                    written = chunk[:count]
                    try:
                        digest.update(written)
                    finally:
                        written.release()
                        written = None
                offset += count
            finally:
                chunk.release()
                chunk = None
    finally:
        if view is not None:
            view.release()
        data = view = chunk = written = None


def _read_exact(stream, data):
    view = chunk = None
    try:
        view = memoryview(data).cast("B")
        offset = 0
        while offset < len(view):
            chunk = view[offset:offset + KV_STORE_CHUNK_BYTES]
            try:
                count = stream.readinto(chunk)
                if type(count) is not int or not 0 < count <= len(chunk):
                    raise ValueError("KV backing file is truncated or the read made no progress")
                offset += count
            finally:
                chunk.release()
                chunk = None
    finally:
        if view is not None:
            view.release()
        data = view = chunk = None


@dataclass(frozen=True)
class KVPageRef:
    path: Path
    descriptor: TieredKVPage
    extent_bytes: int
    checksum: str
    file_bytes: int

    @property
    def payload_offset(self):
        return self.file_bytes - self.extent_bytes

    @property
    def page_index(self):
        return self.descriptor.page_index

    @property
    def logical_start(self):
        return self.descriptor.logical_start

    @property
    def valid_tokens(self):
        return self.descriptor.valid_tokens

    @property
    def codec(self):
        return self.descriptor.codec


class KVPageStore:
    """An exclusive scratch directory; only issued references can be loaded.

    close() unlinks this instance's files and removes its directory if empty.
    Neither the parent nor an unrelated file is removed. Store instances are
    synchronous and must not be used concurrently.
    """
    def __init__(self, parent_path, *, identity):
        if not isinstance(identity, str) or not identity or len(identity) > 4096:
            raise ValueError("KV store identity must be a nonempty string of at most 4096 characters")
        # Preflight bounded encoding, including escaped non-ASCII identities.
        if len(json.dumps(identity, ensure_ascii=True)) > 4096:
            raise ValueError("Encoded KV store identity exceeds 4096 bytes")
        parent = Path(parent_path)
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="nexa-kv-", dir=parent)).resolve()
        descriptor = None
        # Windows does not support opening directory handles this way or dir_fd.
        # Its fallback retains the private namespace and post-open regular-file
        # validation, without claiming the same resistance to pathname races.
        if all(function in os.supports_dir_fd for function in (os.open, os.unlink, os.rename)):
            try:
                descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                     | getattr(os, "O_CLOEXEC", 0))
            except BaseException:
                directory.rmdir()
                raise
        self._identity = identity
        self._directory = directory
        self._directory_fd = descriptor
        self._refs = {}
        # One count per published file and one per session holding this store,
        # so a derived sequence can read a page its parent already closed over.
        self._ref_counts = {}
        self._holders = 1
        self._pending_cleanup = {}
        self._closed = False

    @property
    def directory(self):
        return self._directory

    @property
    def identity(self):
        return self._identity

    @property
    def closed(self):
        return self._closed

    def _check_open(self):
        if self._closed:
            raise ValueError("KV backing store is closed")

    def _validate_ref(self, ref):
        self._check_open()
        if (not isinstance(ref, KVPageRef) or not isinstance(ref.path, Path)
                or ref.path.parent != self.directory or self._refs.get(ref.path.name) is not ref):
            raise ValueError("KV page reference does not belong to this backing store")
        return ref.path.name

    @property
    def holders(self):
        return self._holders

    def references(self, ref):
        """How many owners hold this page; zero once it is fully released."""
        return self._ref_counts.get(ref.path.name, 0) if self.contains(ref) else 0

    def open_shared(self):
        """Take another hold on this store, for a sequence reading its pages."""
        self._check_open()
        self._holders += 1
        return self

    def retain(self, ref):
        """Share one published page with another owner."""
        name = self._validate_ref(ref)
        self._ref_counts[name] += 1
        return ref

    def contains(self, ref):
        """Whether this exact reference is still live; safe after interrupted removal."""
        return (not self._closed and isinstance(ref, KVPageRef) and isinstance(ref.path, Path)
                and ref.path.parent == self.directory and self._refs.get(ref.path.name) is ref)

    @property
    def _io_directory(self):
        return self._directory_fd if self._directory_fd is not None else self.directory

    def _unlink(self, name):
        if self._directory_fd is None:
            (self.directory / name).unlink()
        else:
            os.unlink(name, dir_fd=self._directory_fd)

    def _publish(self, temporary, name):
        try:
            self._stat(name)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("KV backing destination already exists")
        if self._directory_fd is None:
            os.replace(self.directory / temporary, self.directory / name)
        else:
            os.replace(temporary, name, src_dir_fd=self._directory_fd, dst_dir_fd=self._directory_fd)

    def _stat(self, name):
        if self._directory_fd is None:
            return (self.directory / name).lstat()
        return os.stat(name, dir_fd=self._directory_fd, follow_symlinks=False)

    def _cleanup_pending_file(self, name):
        """Remove a failed transaction's own inode, retaining failed cleanup."""
        expected_inode = self._pending_cleanup[name]
        try:
            if expected_inode is not None:
                info = self._stat(name)
                if (info.st_dev, info.st_ino) != expected_inode:
                    # The future publish path can already contain an unrelated
                    # file, or an owned path may have been replaced externally.
                    del self._pending_cleanup[name]
                    return
            self._unlink(name)
        except FileNotFoundError:
            pass
        del self._pending_cleanup[name]

    def write_page(self, descriptor, source_address, extent_bytes):
        self._check_open()
        _validate_descriptor(descriptor, extent_bytes)
        _pointer(source_address, extent_bytes)
        metadata = _metadata(self.identity, descriptor, extent_bytes, "0" * 64)
        file_bytes = KV_STORE_HEADER.size + len(metadata) + extent_bytes
        _integer(file_bytes, "backing file_bytes")
        token = uuid.uuid4().hex
        temporary, name = f"temporary-{token}", f"page-{token}.kvp"
        created = published = committed = False
        payload = owned_inode = None
        try:
            stream = _open_file(self._io_directory, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            created = True
            with stream:
                info = os.fstat(stream.fileno())
                owned_inode = (info.st_dev, info.st_ino)
                # Reserve header/index positions without writing placeholder
                # bytes. Each successfully published file byte is written once.
                stream.seek(KV_STORE_HEADER.size + len(metadata))
                digest = hashlib.sha256()
                for offset in range(0, extent_bytes, KV_STORE_CHUNK_BYTES):
                    count = min(KV_STORE_CHUNK_BYTES, extent_bytes - offset)
                    payload = (ctypes.c_uint8 * count).from_address(source_address + offset)
                    try:
                        _write_all(stream, payload, digest)
                    finally:
                        payload = None
                checksum = digest.hexdigest()
                metadata = _metadata(self.identity, descriptor, extent_bytes, checksum)
                stream.seek(0)
                _write_all(stream, _header(metadata, extent_bytes))
                _write_all(stream, metadata)
                os.fsync(stream.fileno())
            ref = KVPageRef(self.directory / name, descriptor, extent_bytes, checksum, file_bytes)
            self._publish(temporary, name)
            published = True
            self._refs[name] = ref
            self._ref_counts[name] = 1
            committed = True
            return ref
        finally:
            metadata = payload = None
            if not committed:
                self._refs.pop(name, None)
                self._ref_counts.pop(name, None)
                if created:
                    self._pending_cleanup[temporary] = owned_inode
                    # A cancellation can arrive after rename but before its
                    # return. Track the possible destination by the source
                    # inode; close can retry even when stat/unlink now fails.
                    if owned_inode is not None:
                        self._pending_cleanup[name] = owned_inode
                    for leaf in (temporary, name):
                        if leaf not in self._pending_cleanup:
                            continue
                        try:
                            self._cleanup_pending_file(leaf)
                        except OSError:
                            # Preserve the original write failure/cancellation
                            # while retaining ownership for a later close().
                            pass

    def read_page(self, ref, destination_address, capacity):
        name = self._validate_ref(ref)
        _integer(capacity, "reload capacity")
        if capacity < ref.extent_bytes:
            raise ValueError("Reload slot is smaller than the stored KV extent")
        _pointer(destination_address, ref.extent_bytes)
        header = metadata = expected = payload = tail = None
        try:
            with _open_file(self._io_directory, name, os.O_RDONLY) as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size != ref.file_bytes:
                    raise ValueError("KV backing file size or type differs from its reference")
                header = bytearray(KV_STORE_HEADER.size)
                _read_exact(stream, header)
                magic, version, flags, metadata_bytes, extent, metadata_hash = KV_STORE_HEADER.unpack(header)
                if (magic != KV_STORE_MAGIC or version != KV_STORE_VERSION or flags
                        or not 0 < metadata_bytes <= KV_STORE_MAX_METADATA_BYTES
                        or metadata_bytes + KV_STORE_HEADER.size != ref.payload_offset
                        or extent != ref.extent_bytes):
                    raise ValueError("Invalid KV backing header, version or byte counts")
                metadata = bytearray(metadata_bytes)
                _read_exact(stream, metadata)
                if hashlib.sha256(metadata).digest() != metadata_hash:
                    raise ValueError("KV backing metadata checksum mismatch")
                expected = _metadata(self.identity, ref.descriptor, ref.extent_bytes, ref.checksum)
                # Comparing exact canonical bytes validates all metadata without
                # trusting a parser, path, extra field or duplicate JSON key.
                if metadata != expected:
                    raise ValueError("KV backing metadata identity, page descriptor or payload checksum differs")
                metadata = expected = header = None
                digest = hashlib.sha256()
                for offset in range(0, ref.extent_bytes, KV_STORE_CHUNK_BYTES):
                    count = min(KV_STORE_CHUNK_BYTES, ref.extent_bytes - offset)
                    payload = (ctypes.c_uint8 * count).from_address(destination_address + offset)
                    try:
                        _read_exact(stream, payload)
                        digest.update(payload)
                    finally:
                        payload = None
                tail = bytearray(1)
                if stream.readinto(tail) != 0 or os.fstat(stream.fileno()).st_size != ref.file_bytes:
                    raise ValueError("KV backing file contains trailing bytes or changed size")
                if digest.hexdigest() != ref.checksum:
                    raise ValueError("KV backing payload checksum mismatch")
            return ref.file_bytes
        finally:
            header = metadata = expected = payload = tail = None

    def remove(self, ref):
        """Drop one owner's hold; the file survives while others still hold it."""
        name = self._validate_ref(ref)
        if self._ref_counts.get(name, 1) > 1:
            self._ref_counts[name] -= 1
            return
        try:
            self._unlink(name)
        except FileNotFoundError:
            pass
        del self._refs[name]
        self._ref_counts.pop(name, None)

    def close(self):
        """Release this hold; the last one removes the files and the directory."""
        if self._closed:
            return
        if self._holders > 1:
            self._holders -= 1
            return
        failure = None
        for name in tuple(self._refs):
            try:
                self._ref_counts.pop(name, None)
                self._unlink(name)
            except FileNotFoundError:
                pass
            except OSError as error:
                failure = failure or error
                continue
            del self._refs[name]
        for name in tuple(self._pending_cleanup):
            try:
                self._cleanup_pending_file(name)
            except OSError as error:
                failure = failure or error
        if failure is not None:
            raise failure
        try:
            self.directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError as error:
            if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
        if self._directory_fd is not None:
            os.close(self._directory_fd)
        self._directory_fd = None
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
