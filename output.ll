; ModuleID = "nexalang_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

define void @"main"()
{
entry:
  %"a" = alloca i32
  store i32 10, i32* %"a"
  %"a.1" = load i32, i32* %"a"
  %"eqtmp" = icmp eq i32 %"a.1", 10
  br i1 %"eqtmp", label %"then", label %"else"
then:
  %".4" = bitcast [4 x i8]* @"fmt_s" to i8*
  %".5" = bitcast [8 x i8]* @"str" to i8*
  %".6" = call i32 (i8*, ...) @"printf"(i8* %".4", i8* %".5")
  br label %"ifcont"
else:
  %".8" = bitcast [4 x i8]* @"fmt_s.1" to i8*
  %".9" = bitcast [12 x i8]* @"str.1" to i8*
  %".10" = call i32 (i8*, ...) @"printf"(i8* %".8", i8* %".9")
  br label %"ifcont"
ifcont:
  %"a.2" = load i32, i32* %"a"
  %"eqtmp.1" = icmp eq i32 %"a.2", 5
  br i1 %"eqtmp.1", label %"then.1", label %"else.1"
then.1:
  %".13" = bitcast [4 x i8]* @"fmt_s.2" to i8*
  %".14" = bitcast [7 x i8]* @"str.2" to i8*
  %".15" = call i32 (i8*, ...) @"printf"(i8* %".13", i8* %".14")
  br label %"ifcont.1"
else.1:
  %".17" = bitcast [4 x i8]* @"fmt_s.3" to i8*
  %".18" = bitcast [11 x i8]* @"str.3" to i8*
  %".19" = call i32 (i8*, ...) @"printf"(i8* %".17", i8* %".18")
  br label %"ifcont.1"
ifcont.1:
  ret void
}

@"str" = internal constant [8 x i8] c"a is 10\00"
@"fmt_s" = internal constant [4 x i8] c"%s\0a\00"
@"str.1" = internal constant [12 x i8] c"a is NOT 10\00"
@"fmt_s.1" = internal constant [4 x i8] c"%s\0a\00"
@"str.2" = internal constant [7 x i8] c"a is 5\00"
@"fmt_s.2" = internal constant [4 x i8] c"%s\0a\00"
@"str.3" = internal constant [11 x i8] c"a is NOT 5\00"
@"fmt_s.3" = internal constant [4 x i8] c"%s\0a\00"