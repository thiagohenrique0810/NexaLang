; ModuleID = "nexalang_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

declare void @"exit"(i32 %".1")

define void @"main"()
{
entry:
  %".2" = bitcast [4 x i8]* @"fmt_s" to i8*
  %".3" = bitcast [17 x i8]* @"str" to i8*
  %".4" = call i32 (i8*, ...) @"printf"(i8* %".2", i8* %".3")
  %".5" = insertvalue {i32, [4 x i8]} undef, i32 0, 0
  %".6" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %".5", {i32, [4 x i8]}* %".6"
  %".8" = getelementptr {i32, [4 x i8]}, {i32, [4 x i8]}* %".6", i32 0, i32 1
  %".9" = bitcast [4 x i8]* %".8" to i32*
  store i32 42, i32* %".9"
  %".11" = load {i32, [4 x i8]}, {i32, [4 x i8]}* %".6"
  %"res" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %".11", {i32, [4 x i8]}* %"res"
  %"res.1" = load {i32, [4 x i8]}, {i32, [4 x i8]}* %"res"
  %".13" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %"res.1", {i32, [4 x i8]}* %".13"
  %".15" = extractvalue {i32, [4 x i8]} %"res.1", 0
  switch i32 %".15", label %"match_merge" [i32 0, label %"case_Ok" i32 1, label %"case_Err"]
match_merge:
  %".32" = insertvalue {i32, [4 x i8]} undef, i32 1, 0
  %".33" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %".32", {i32, [4 x i8]}* %".33"
  %".35" = getelementptr {i32, [4 x i8]}, {i32, [4 x i8]}* %".33", i32 0, i32 1
  %".36" = bitcast [4 x i8]* %".35" to i32*
  store i32 500, i32* %".36"
  %".38" = load {i32, [4 x i8]}, {i32, [4 x i8]}* %".33"
  %"err" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %".38", {i32, [4 x i8]}* %"err"
  %"err.1" = load {i32, [4 x i8]}, {i32, [4 x i8]}* %"err"
  %".40" = alloca {i32, [4 x i8]}
  store {i32, [4 x i8]} %"err.1", {i32, [4 x i8]}* %".40"
  %".42" = extractvalue {i32, [4 x i8]} %"err.1", 0
  switch i32 %".42", label %"match_merge.1" [i32 0, label %"case_Ok.1" i32 1, label %"case_Err.1"]
case_Ok:
  %".17" = getelementptr {i32, [4 x i8]}, {i32, [4 x i8]}* %".13", i32 0, i32 1
  %".18" = bitcast [4 x i8]* %".17" to i32*
  %"val" = load i32, i32* %".18"
  %".19" = bitcast [4 x i8]* @"fmt_d" to i8*
  %".20" = call i32 (i8*, ...) @"printf"(i8* %".19", i32 %"val")
  br label %"match_merge"
case_Err:
  %".22" = getelementptr {i32, [4 x i8]}, {i32, [4 x i8]}* %".13", i32 0, i32 1
  %".23" = bitcast [4 x i8]* %".22" to i32*
  %".24" = bitcast [8 x i8]* @"panic_prefix" to i8*
  %".25" = bitcast [4 x i8]* @"fmt_s.1" to i8*
  %".26" = bitcast [8 x i8]* @"panic_prefix" to i8*
  %".27" = call i32 (i8*, ...) @"printf"(i8* %".25", i8* %".26")
  %".28" = bitcast [13 x i8]* @"str.1" to i8*
  %".29" = call i32 (i8*, ...) @"printf"(i8* %".25", i8* %".28")
  call void @"exit"(i32 1)
  unreachable
match_merge.1:
  %".59" = bitcast [4 x i8]* @"fmt_s.3" to i8*
  %".60" = bitcast [6 x i8]* @"str.3" to i8*
  %".61" = call i32 (i8*, ...) @"printf"(i8* %".59", i8* %".60")
  ret void
case_Ok.1:
  %".44" = getelementptr {i32, [4 x i8]}, {i32, [4 x i8]}* %".40", i32 0, i32 1
  %".45" = bitcast [4 x i8]* %".44" to i32*
  %".46" = bitcast [8 x i8]* @"panic_prefix.1" to i8*
  %".47" = bitcast [4 x i8]* @"fmt_s.2" to i8*
  %".48" = bitcast [8 x i8]* @"panic_prefix.1" to i8*
  %".49" = call i32 (i8*, ...) @"printf"(i8* %".47", i8* %".48")
  %".50" = bitcast [14 x i8]* @"str.2" to i8*
  %".51" = call i32 (i8*, ...) @"printf"(i8* %".47", i8* %".50")
  call void @"exit"(i32 1)
  unreachable
case_Err.1:
  %".54" = getelementptr {i32, [4 x i8]}, {i32, [4 x i8]}* %".40", i32 0, i32 1
  %".55" = bitcast [4 x i8]* %".54" to i32*
  %"code" = load i32, i32* %".55"
  %".56" = bitcast [4 x i8]* @"fmt_d.1" to i8*
  %".57" = call i32 (i8*, ...) @"printf"(i8* %".56", i32 %"code")
  br label %"match_merge.1"
}

@"str" = internal constant [17 x i8] c"Testing Enums...\00"
@"fmt_s" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_d" = internal constant [4 x i8] c"%d\0a\00"
@"str.1" = internal constant [13 x i8] c"Should be Ok\00"
@"panic_prefix" = internal constant [8 x i8] c"PANIC: \00"
@"fmt_s.1" = internal constant [4 x i8] c"%s\0a\00"
@"str.2" = internal constant [14 x i8] c"Should be Err\00"
@"panic_prefix.1" = internal constant [8 x i8] c"PANIC: \00"
@"fmt_s.2" = internal constant [4 x i8] c"%s\0a\00"
@"fmt_d.1" = internal constant [4 x i8] c"%d\0a\00"
@"str.3" = internal constant [6 x i8] c"Done.\00"
@"fmt_s.3" = internal constant [4 x i8] c"%s\0a\00"