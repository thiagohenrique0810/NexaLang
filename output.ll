; ModuleID = "nexalang_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

declare void @"exit"(i32 %".1")

define void @"main"()
{
entry:
  %".2" = bitcast [4 x i8]* @"fmt_s" to i8*
  %".3" = bitcast [18 x i8]* @"str" to i8*
  %".4" = call i32 (i8*, ...) @"printf"(i8* %".2", i8* %".3")
  %"eqtmp" = icmp eq i32 1, 1
  %"assert_cond" = icmp ne i1 %"eqtmp", 0
  br i1 %"assert_cond", label %"assert_cont", label %"assert_fail"
assert_fail:
  %".6" = bitcast [4 x i8]* @"fmt_s.1" to i8*
  %".7" = bitcast [19 x i8]* @"assert_prefix" to i8*
  %".8" = call i32 (i8*, ...) @"printf"(i8* %".6", i8* %".7")
  %".9" = bitcast [15 x i8]* @"str.1" to i8*
  %".10" = call i32 (i8*, ...) @"printf"(i8* %".6", i8* %".9")
  call void @"exit"(i32 1)
  unreachable
assert_cont:
  %".13" = bitcast [4 x i8]* @"fmt_s.2" to i8*
  %".14" = bitcast [20 x i8]* @"str.2" to i8*
  %".15" = call i32 (i8*, ...) @"printf"(i8* %".13", i8* %".14")
  %"x" = alloca i32
  store i32 10, i32* %"x"
  %"x.1" = load i32, i32* %"x"
  %"eqtmp.1" = icmp eq i32 %"x.1", 5
  %"assert_cond.1" = icmp ne i1 %"eqtmp.1", 0
  br i1 %"assert_cond.1", label %"assert_cont.1", label %"assert_fail.1"
assert_fail.1:
  %".18" = bitcast [4 x i8]* @"fmt_s.3" to i8*
  %".19" = bitcast [19 x i8]* @"assert_prefix.1" to i8*
  %".20" = call i32 (i8*, ...) @"printf"(i8* %".18", i8* %".19")
  %".21" = bitcast [14 x i8]* @"str.3" to i8*
  %".22" = call i32 (i8*, ...) @"printf"(i8* %".18", i8* %".21")
  call void @"exit"(i32 1)
  unreachable
assert_cont.1:
  %".25" = bitcast [4 x i8]* @"fmt_s.4" to i8*
  %".26" = bitcast [22 x i8]* @"str.4" to i8*
  %".27" = call i32 (i8*, ...) @"printf"(i8* %".25", i8* %".26")
  ret void
}

@"str" = internal constant [18 x i8] c"Testing Assert...\00"
@"fmt_s" = internal constant [4 x i8] c"%s\0a\00"
@"str.1" = internal constant [15 x i8] c"Math is broken\00"
@"assert_prefix" = internal constant [19 x i8] c"ASSERTION FAILED: \00"
@"fmt_s.1" = internal constant [4 x i8] c"%s\0a\00"
@"str.2" = internal constant [20 x i8] c"First assert passed\00"
@"fmt_s.2" = internal constant [4 x i8] c"%s\0a\00"
@"str.3" = internal constant [14 x i8] c"x should be 5\00"
@"assert_prefix.1" = internal constant [19 x i8] c"ASSERTION FAILED: \00"
@"fmt_s.3" = internal constant [4 x i8] c"%s\0a\00"
@"str.4" = internal constant [22 x i8] c"Should not be reached\00"
@"fmt_s.4" = internal constant [4 x i8] c"%s\0a\00"