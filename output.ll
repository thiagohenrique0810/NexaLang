; ModuleID = "nexalang_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

declare void @"exit"(i32 %".1")

declare i8* @"malloc"(i32 %".1")

declare void @"free"(i8* %".1")

declare i8* @"realloc"(i8* %".1", i32 %".2")

declare void @"llvm.memcpy.p0i8.p0i8.i32"(i8* %".1", i8* %".2", i32 %".3", i1 %".4")

define void @"String_drop"({i8*, i32, i32} %"s")
{
entry:
  %"s.1" = alloca {i8*, i32, i32}
  store {i8*, i32, i32} %"s", {i8*, i32, i32}* %"s.1"
  %".4" = bitcast [20 x i8]* @"str" to i8*
  %".5" = bitcast [4 x i8]* @"fmt_s" to i8*
  %".6" = call i32 (i8*, ...) @"printf"(i8* %".5", i8* %".4")
  %"s.2" = load {i8*, i32, i32}, {i8*, i32, i32}* %"s.1"
  %".7" = extractvalue {i8*, i32, i32} %"s.2", 0
  %".8" = bitcast [4 x i8]* @"fmt_s.1" to i8*
  %".9" = call i32 (i8*, ...) @"printf"(i8* %".8", i8* %".7")
  ret void
}

define {i8*, i32, i32} @"string_from"(i8* %"s")
{
entry:
  %"s.1" = alloca i8*
  store i8* %"s", i8** %"s.1"
  %"s.2" = load i8*, i8** %"s.1"
  %".4" = insertvalue {i8*, i32, i32} undef, i8* %"s.2", 0
  %".5" = insertvalue {i8*, i32, i32} %".4", i32 0, 1
  %".6" = insertvalue {i8*, i32, i32} %".5", i32 0, 2
  ret {i8*, i32, i32} %".6"
}

define void @"test_scope"()
{
entry:
  %".2" = bitcast [15 x i8]* @"str.1" to i8*
  %".3" = bitcast [4 x i8]* @"fmt_s.2" to i8*
  %".4" = call i32 (i8*, ...) @"printf"(i8* %".3", i8* %".2")
  %".5" = bitcast [6 x i8]* @"str.2" to i8*
  %".6" = call {i8*, i32, i32} @"string_from"(i8* %".5")
  %"s1" = alloca {i8*, i32, i32}
  store {i8*, i32, i32} %".6", {i8*, i32, i32}* %"s1"
  store {i8*, i32, i32} %".6", {i8*, i32, i32}* %"s1"
  %".9" = bitcast [13 x i8]* @"str.3" to i8*
  %".10" = bitcast [4 x i8]* @"fmt_s.3" to i8*
  %".11" = call i32 (i8*, ...) @"printf"(i8* %".10", i8* %".9")
  ret void
}

define void @"main"()
{
entry:
  %".2" = bitcast [6 x i8]* @"str.4" to i8*
  %".3" = call {i8*, i32, i32} @"string_from"(i8* %".2")
  %"s2" = alloca {i8*, i32, i32}
  store {i8*, i32, i32} %".3", {i8*, i32, i32}* %"s2"
  store {i8*, i32, i32} %".3", {i8*, i32, i32}* %"s2"
  call void @"test_scope"()
  %".7" = bitcast [13 x i8]* @"str.5" to i8*
  %".8" = bitcast [4 x i8]* @"fmt_s.4" to i8*
  %".9" = call i32 (i8*, ...) @"printf"(i8* %".8", i8* %".7")
  ret void
}

@"str" = internal constant [20 x i8] c"Dropping String... \00"
@"fmt_s" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_s.1" = internal constant [4 x i8] c"%s\0a\00"
@"str.1" = internal constant [15 x i8] c"Entering scope\00"
@"fmt_s.2" = internal constant [4 x i8] c"%s\0a\00"
@"str.2" = internal constant [6 x i8] c"Inner\00"
@"str.3" = internal constant [13 x i8] c"Inside scope\00"
@"fmt_s.3" = internal constant [4 x i8] c"%s\0a\00"
@"str.4" = internal constant [6 x i8] c"Outer\00"
@"str.5" = internal constant [13 x i8] c"Back in main\00"
@"fmt_s.4" = internal constant [4 x i8] c"%s\0a\00"