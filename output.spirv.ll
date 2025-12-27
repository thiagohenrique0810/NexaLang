; ModuleID = "nexalang_module"
target triple = "spir64-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

declare void @"exit"(i32 %".1")

declare i8* @"malloc"(i32 %".1")

declare void @"free"(i8* %".1")

declare i8* @"realloc"(i8* %".1", i32 %".2")

declare void @"llvm.memcpy.p0i8.p0i8.i32"(i8* %".1", i8* %".2", i32 %".3", i1 %".4")

@"__gpu_global_id" = internal global i32 0
@"__spirv_BuiltInGlobalInvocationId_x" = external global i32
define {i8*, i32, i32} @"Arena_new"()
{
entry:
  %".2" = call i8* @"malloc"(i32 4096)
  %".3" = insertvalue {i8*, i32, i32} undef, i8* %".2", 0
  %".4" = insertvalue {i8*, i32, i32} %".3", i32 0, 1
  %".5" = insertvalue {i8*, i32, i32} %".4", i32 4096, 2
  ret {i8*, i32, i32} %".5"
}

define void @"Arena_drop"({i8*, i32, i32} %".1")
{
entry:
  %".3" = extractvalue {i8*, i32, i32} %".1", 0
  call void @"free"(i8* %".3")
  ret void
}

define i8* @"Arena_alloc"({i8*, i32, i32}* %".1", i32 %".2")
{
entry:
  %".4" = getelementptr {i8*, i32, i32}, {i8*, i32, i32}* %".1", i32 0, i32 0
  %".5" = getelementptr {i8*, i32, i32}, {i8*, i32, i32}* %".1", i32 0, i32 1
  %".6" = getelementptr {i8*, i32, i32}, {i8*, i32, i32}* %".1", i32 0, i32 2
  %".7" = load i8*, i8** %".4"
  %".8" = load i32, i32* %".5"
  %".9" = load i32, i32* %".6"
  %".10" = add i32 %".8", %".2"
  %".11" = icmp ugt i32 %".10", %".9"
  br i1 %".11", label %"entry.if", label %"entry.endif"
entry.if:
  call void @"exit"(i32 1)
  br label %"entry.endif"
entry.endif:
  %".15" = ptrtoint i8* %".7" to i64
  %".16" = zext i32 %".8" to i64
  %".17" = add i64 %".15", %".16"
  %".18" = inttoptr i64 %".17" to i8*
  store i32 %".10", i32* %".5"
  ret i8* %".18"
}

define spir_kernel void @"compute"()
{
entry:
  %"spirv_global_id_x" = load i32, i32* @"__spirv_BuiltInGlobalInvocationId_x"
  %"id" = alloca i32
  store i32 %"spirv_global_id_x", i32* %"id"
  store i32 %"spirv_global_id_x", i32* %"id"
  %".4" = bitcast [6 x i8]* @"str" to i8*
  %".5" = bitcast [4 x i8]* @"fmt_s" to i8*
  %".6" = call i32 (i8*, ...) @"printf"(i8* %".5", i8* %".4")
  %"id.1" = load i32, i32* %"id"
  %".7" = bitcast [4 x i8]* @"fmt_d" to i8*
  %".8" = call i32 (i8*, ...) @"printf"(i8* %".7", i32 %"id.1")
  ret void
}

define void @"main"()
{
entry:
  %".2" = bitcast [38 x i8]* @"str.1" to i8*
  %".3" = bitcast [4 x i8]* @"fmt_s.1" to i8*
  %".4" = call i32 (i8*, ...) @"printf"(i8* %".3", i8* %".2")
  %"gpu_i" = alloca i32
  store i32 0, i32* %"gpu_i"
  br label %"gpu_dispatch_cond"
gpu_dispatch_cond:
  %"gpu_i_val" = load i32, i32* %"gpu_i"
  %"gpu_cond" = icmp slt i32 %"gpu_i_val", 4
  br i1 %"gpu_cond", label %"gpu_dispatch_body", label %"gpu_dispatch_end"
gpu_dispatch_body:
  store i32 %"gpu_i_val", i32* @"__gpu_global_id"
  call spir_kernel void @"compute"()
  %"gpu_i_inc" = add i32 %"gpu_i_val", 1
  store i32 %"gpu_i_inc", i32* %"gpu_i"
  br label %"gpu_dispatch_cond"
gpu_dispatch_end:
  ret void
}

@"str" = internal constant [6 x i8] c"ID = \00"
@"fmt_s" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_d" = internal constant [4 x i8] c"%d\0a\00"
@"str.1" = internal constant [38 x i8] c"Dispatching kernel (mock CPU loop)...\00"
@"fmt_s.1" = internal constant [4 x i8] c"%s\0a\00"