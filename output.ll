; ModuleID = "nexalang_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

define void @"main"()
{
entry:
  %".2" = insertvalue [3 x i32] undef, i32 10, 0
  %".3" = insertvalue [3 x i32] %".2", i32 20, 1
  %".4" = insertvalue [3 x i32] %".3", i32 30, 2
  %"arr" = alloca [3 x i32]
  store [3 x i32] %".4", [3 x i32]* %"arr"
  %".6" = bitcast [4 x i8]* @"fmt_s" to i8*
  %".7" = bitcast [11 x i8]* @"str" to i8*
  %".8" = call i32 (i8*, ...) @"printf"(i8* %".6", i8* %".7")
  %".9" = getelementptr [3 x i32], [3 x i32]* %"arr", i32 0, i32 0
  %".10" = load i32, i32* %".9"
  %".11" = bitcast [4 x i8]* @"fmt_d" to i8*
  %".12" = call i32 (i8*, ...) @"printf"(i8* %".11", i32 %".10")
  %".13" = bitcast [4 x i8]* @"fmt_s.1" to i8*
  %".14" = bitcast [11 x i8]* @"str.1" to i8*
  %".15" = call i32 (i8*, ...) @"printf"(i8* %".13", i8* %".14")
  %".16" = getelementptr [3 x i32], [3 x i32]* %"arr", i32 0, i32 1
  %".17" = load i32, i32* %".16"
  %".18" = bitcast [4 x i8]* @"fmt_d.1" to i8*
  %".19" = call i32 (i8*, ...) @"printf"(i8* %".18", i32 %".17")
  %"i" = alloca i32
  store i32 2, i32* %"i"
  %".21" = bitcast [4 x i8]* @"fmt_s.2" to i8*
  %".22" = bitcast [11 x i8]* @"str.2" to i8*
  %".23" = call i32 (i8*, ...) @"printf"(i8* %".21", i8* %".22")
  %"i.1" = load i32, i32* %"i"
  %".24" = getelementptr [3 x i32], [3 x i32]* %"arr", i32 0, i32 %"i.1"
  %".25" = load i32, i32* %".24"
  %".26" = bitcast [4 x i8]* @"fmt_d.2" to i8*
  %".27" = call i32 (i8*, ...) @"printf"(i8* %".26", i32 %".25")
  ret void
}

@"str" = internal constant [11 x i8] c"Array[0]: \00"
@"fmt_s" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_d" = internal constant [4 x i8] c"%d\0a\00"
@"str.1" = internal constant [11 x i8] c"Array[1]: \00"
@"fmt_s.1" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_d.1" = internal constant [4 x i8] c"%d\0a\00"
@"str.2" = internal constant [11 x i8] c"Array[i]: \00"
@"fmt_s.2" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_d.2" = internal constant [4 x i8] c"%d\0a\00"