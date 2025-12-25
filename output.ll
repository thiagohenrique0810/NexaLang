; ModuleID = "nexalang_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

define void @"main"()
{
entry:
  %"i" = alloca i32
  store i32 5, i32* %"i"
  br label %"whilecond"
whilecond:
  %"i.1" = load i32, i32* %"i"
  %"loopcond" = icmp ne i32 %"i.1", 0
  br i1 %"loopcond", label %"whileloop", label %"whileend"
whileloop:
  %"i.2" = load i32, i32* %"i"
  %".5" = bitcast [4 x i8]* @"fmt_d" to i8*
  %".6" = call i32 (i8*, ...) @"printf"(i8* %".5", i32 %"i.2")
  %"i.3" = load i32, i32* %"i"
  %"subtmp" = sub i32 %"i.3", 1
  store i32 %"subtmp", i32* %"i"
  br label %"whilecond"
whileend:
  %".9" = bitcast [4 x i8]* @"fmt_s" to i8*
  %".10" = bitcast [14 x i8]* @"str" to i8*
  %".11" = call i32 (i8*, ...) @"printf"(i8* %".9", i8* %".10")
  ret void
}

@"fmt_d" = internal constant [4 x i8] c"%d\0a\00"
@"str" = internal constant [14 x i8] c"Loop finished\00"
@"fmt_s" = internal constant [4 x i8] c"%s\0a\00"