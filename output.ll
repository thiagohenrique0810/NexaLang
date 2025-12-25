; ModuleID = "nexalang_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

declare void @"exit"(i32 %".1")

define i32 @"unwrap_or"({i32, [4 x i8]} %"opt", i32 %"default")
{
entry:
  %"opt.1" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %"opt", {i32, [4 x i8]}* %"opt.1"
  %"default.1" = alloca i32
  store i32 %"default", i32* %"default.1"
  %"opt.2" = load {i32, [4 x i8]}, {i32, [4 x i8]}* %"opt.1"
  %".6" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %"opt.2", {i32, [4 x i8]}* %".6"
  %".8" = extractvalue {i32, [4 x i8]} %"opt.2", 0
  switch i32 %".8", label %"match_merge" [i32 0, label %"case_Some" i32 1, label %"case_None"]
match_merge:
  ret i32 0
case_Some:
  %".10" = getelementptr {i32, [4 x i8]}, {i32, [4 x i8]}* %".6", i32 0, i32 1
  %".11" = bitcast [4 x i8]* %".10" to i32*
  %"val" = load i32, i32* %".11"
  br label %"match_merge"
case_None:
  %"default.2" = load i32, i32* %"default.1"
  br label %"match_merge"
}

define void @"main"()
{
entry:
  %".2" = bitcast [4 x i8]* @"fmt_s" to i8*
  %".3" = bitcast [37 x i8]* @"str" to i8*
  %".4" = call i32 (i8*, ...) @"printf"(i8* %".2", i8* %".3")
  %".5" = insertvalue {i32, [4 x i8]} undef, i32 0, 0
  %".6" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %".5", {i32, [4 x i8]}* %".6"
  %".8" = getelementptr {i32, [4 x i8]}, {i32, [4 x i8]}* %".6", i32 0, i32 1
  %".9" = bitcast [4 x i8]* %".8" to i32*
  store i32 42, i32* %".9"
  %".11" = load {i32, [4 x i8]}, {i32, [4 x i8]}* %".6"
  %"some" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %".11", {i32, [4 x i8]}* %"some"
  %".13" = insertvalue {i32, [4 x i8]} undef, i32 1, 0
  %"none" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %".13", {i32, [4 x i8]}* %"none"
  %"some.1" = load {i32, [4 x i8]}, {i32, [4 x i8]}* %"some"
  %".15" = call i32 @"unwrap_or"({i32, [4 x i8]} %"some.1", i32 0)
  %"v1" = alloca i32
  store i32 %".15", i32* %"v1"
  %"v1.1" = load i32, i32* %"v1"
  %".17" = bitcast [4 x i8]* @"fmt_d" to i8*
  %".18" = call i32 (i8*, ...) @"printf"(i8* %".17", i32 %"v1.1")
  %"none.1" = load {i32, [4 x i8]}, {i32, [4 x i8]}* %"none"
  %".19" = call i32 @"unwrap_or"({i32, [4 x i8]} %"none.1", i32 0)
  %"v2" = alloca i32
  store i32 %".19", i32* %"v2"
  %"v2.1" = load i32, i32* %"v2"
  %".21" = bitcast [4 x i8]* @"fmt_d.1" to i8*
  %".22" = call i32 (i8*, ...) @"printf"(i8* %".21", i32 %"v2.1")
  %".23" = insertvalue {i32, [8 x i8]} undef, i32 0, 0
  %".24" = alloca {i32, [8 x i8]}
  store {i32, [8 x i8]} %".23", {i32, [8 x i8]}* %".24"
  %".26" = getelementptr {i32, [8 x i8]}, {i32, [8 x i8]}* %".24", i32 0, i32 1
  %".27" = bitcast [8 x i8]* %".26" to i32*
  store i32 100, i32* %".27"
  %".29" = load {i32, [8 x i8]}, {i32, [8 x i8]}* %".24"
  %"ok" = alloca {i32, [8 x i8]}
  store {i32, [8 x i8]} %".29", {i32, [8 x i8]}* %"ok"
  %".31" = insertvalue {i32, [8 x i8]} undef, i32 1, 0
  %".32" = alloca {i32, [8 x i8]}
  store {i32, [8 x i8]} %".31", {i32, [8 x i8]}* %".32"
  %".34" = getelementptr {i32, [8 x i8]}, {i32, [8 x i8]}* %".32", i32 0, i32 1
  %".35" = bitcast [8 x i8]* %".34" to [7 x i8]**
  store [7 x i8]* @"str.1", [7 x i8]** %".35"
  %".37" = load {i32, [8 x i8]}, {i32, [8 x i8]}* %".32"
  %"err" = alloca {i32, [8 x i8]}
  store {i32, [8 x i8]} %".37", {i32, [8 x i8]}* %"err"
  %"ok.1" = load {i32, [8 x i8]}, {i32, [8 x i8]}* %"ok"
  %".39" = alloca {i32, [8 x i8]}
  store {i32, [8 x i8]} %"ok.1", {i32, [8 x i8]}* %".39"
  %".41" = extractvalue {i32, [8 x i8]} %"ok.1", 0
  switch i32 %".41", label %"match_merge" [i32 0, label %"case_Ok" i32 1, label %"case_Err"]
match_merge:
  %"err.1" = load {i32, [8 x i8]}, {i32, [8 x i8]}* %"err"
  %".53" = alloca {i32, [8 x i8]}
  store {i32, [8 x i8]} %"err.1", {i32, [8 x i8]}* %".53"
  %".55" = extractvalue {i32, [8 x i8]} %"err.1", 0
  switch i32 %".55", label %"match_merge.1" [i32 0, label %"case_Ok.1" i32 1, label %"case_Err.1"]
case_Ok:
  %".43" = getelementptr {i32, [8 x i8]}, {i32, [8 x i8]}* %".39", i32 0, i32 1
  %".44" = bitcast [8 x i8]* %".43" to i32*
  %"v" = load i32, i32* %".44"
  %".45" = bitcast [4 x i8]* @"fmt_d.2" to i8*
  %".46" = call i32 (i8*, ...) @"printf"(i8* %".45", i32 %"v")
  br label %"match_merge"
case_Err:
  %".48" = getelementptr {i32, [8 x i8]}, {i32, [8 x i8]}* %".39", i32 0, i32 1
  %".49" = bitcast [8 x i8]* %".48" to i8**
  %"e" = load i8*, i8** %".49"
  %".50" = bitcast [4 x i8]* @"fmt_s.1" to i8*
  %".51" = call i32 (i8*, ...) @"printf"(i8* %".50", i8* %"e")
  br label %"match_merge"
match_merge.1:
  %".67" = bitcast [4 x i8]* @"fmt_s.3" to i8*
  %".68" = bitcast [5 x i8]* @"str.2" to i8*
  %".69" = call i32 (i8*, ...) @"printf"(i8* %".67", i8* %".68")
  ret void
case_Ok.1:
  %".57" = getelementptr {i32, [8 x i8]}, {i32, [8 x i8]}* %".53", i32 0, i32 1
  %".58" = bitcast [8 x i8]* %".57" to i32*
  %"v.1" = load i32, i32* %".58"
  %".59" = bitcast [4 x i8]* @"fmt_d.3" to i8*
  %".60" = call i32 (i8*, ...) @"printf"(i8* %".59", i32 %"v.1")
  br label %"match_merge.1"
case_Err.1:
  %".62" = getelementptr {i32, [8 x i8]}, {i32, [8 x i8]}* %".53", i32 0, i32 1
  %".63" = bitcast [8 x i8]* %".62" to i8**
  %"e.1" = load i8*, i8** %".63"
  %".64" = bitcast [4 x i8]* @"fmt_s.2" to i8*
  %".65" = call i32 (i8*, ...) @"printf"(i8* %".64", i8* %"e.1")
  br label %"match_merge.1"
}

@"str" = internal constant [37 x i8] c"Testing Standard Library Patterns...\00"
@"fmt_s" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_d" = internal constant [4 x i8] c"%d\0a\00"
@"fmt_d.1" = internal constant [4 x i8] c"%d\0a\00"
@"str.1" = internal constant [7 x i8] c"Error!\00"
@"fmt_d.2" = internal constant [4 x i8] c"%d\0a\00"
@"fmt_s.1" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_d.3" = internal constant [4 x i8] c"%d\0a\00"
@"fmt_s.2" = internal constant [4 x i8] c"%s\0a\00"
@"str.2" = internal constant [5 x i8] c"Done\00"
@"fmt_s.3" = internal constant [4 x i8] c"%s\0a\00"