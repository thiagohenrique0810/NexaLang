#!/usr/bin/env python3
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('/Users/thiagomac/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.model')

test1 = '<|user|>\nTell me a short story about a cat\n<|assistant|>\n'
ids1 = sp.Encode(test1)
print(f'SP user prompt: {ids1}')
print(f'NXL            : [529, 127, 1792, 127, 65, 13, 87, 514, 592, 263, 3273, 5828, 1048, 263, 6635, 13, 63, 127, 465, 22137, 127, 65, 13]')
pieces1 = [sp.IdToPiece(i) for i in ids1]
print(f'SP pieces: {pieces1}')

# Check specific tokens
print(f'\n--- Key comparisons ---')
print(f'SP piece for |: ID={sp.PieceToId("|")} (should be around 29989)')
print(f'SP piece for <0x7C>: ID={sp.PieceToId("<0x7C>")} (byte token for |)')
print(f'NXL uses ID 127 for | (byte_start=3, 0x7C=124, 3+124=127)')
print(f'SP ID 127: piece={sp.IdToPiece(127)}')
print(f'SP ID 29989: piece={sp.IdToPiece(29989)}')

# Check the prime context
prime = '<|system|>\nYou are a helpful assistant.\n<|user|>\nHello'
prime += '!\n<|assistant|>\nHello'
prime += '! How can I help you?\n'
ids2 = sp.Encode(prime)
print(f'\nSP prime: {ids2}')
print(f'NXL prime: [529, 127, 5205, 127, 65, 13, 3492, 526, 263, 8444, 20255, 49, 13, 63, 127, 1792, 127, 65, 13, 10994, 36, 13, 63, 127, 465, 22137, 127, 65, 13, 10994]')
