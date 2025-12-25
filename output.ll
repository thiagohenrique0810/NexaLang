; ModuleID = "nexalang_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

define spir_kernel void @"compute"()
{
entry:
  %"id" = alloca i32
  store i32 0, i32* %"id"
  %".3" = bitcast [4 x i8]* @"fmt_s" to i8*
  %".4" = bitcast [25 x i8]* @"str" to i8*
  %".5" = call i32 (i8*, ...) @"printf"(i8* %".3", i8* %".4")
  %"id.1" = load i32, i32* %"id"
  %".6" = bitcast [4 x i8]* @"fmt_d" to i8*
  %".7" = call i32 (i8*, ...) @"printf"(i8* %".6", i32 %"id.1")
  ret void
}

@"str" = internal constant [25 x i8] c"Kernel running with ID: \00"
@"fmt_s" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_d" = internal constant [4 x i8] c"%d\0a\00"
define void @"main"()
{
entry:
  %".2" = bitcast [4 x i8]* @"fmt_s.1" to i8*
  %".3" = bitcast [22 x i8]* @"str.1" to i8*
  %".4" = call i32 (i8*, ...) @"printf"(i8* %".2", i8* %".3")
  call spir_kernel void @"compute"()
  ret void
}

@"str.1" = internal constant [22 x i8] c"Dispatching kernel...\00"
@"fmt_s.1" = internal constant [4 x i8] c"%s\0a\00"