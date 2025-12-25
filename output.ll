; ModuleID = "nexalang_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

define void @"main"()
{
entry:
  %".2" = bitcast [4 x i8]* @"fmt" to i8*
  %".3" = bitcast [17 x i8]* @"str" to i8*
  %".4" = call i32 (i8*, ...) @"printf"(i8* %".2", i8* %".3")
  ret void
}

@"str" = internal constant [17 x i8] c"Hello, NexaLang!\00"
@"fmt" = internal constant [4 x i8] c"%s\0a\00"