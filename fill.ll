; ModuleID = "nexalang_module"
target triple = "spir64-unknown-unknown"
target datalayout = ""

declare i32 @"printf"(i8* %".1", ...)

declare void @"exit"(i32 %".1")

declare i8* @"malloc"(i32 %".1")

declare void @"free"(i8* %".1")

declare i8* @"realloc"(i8* %".1", i32 %".2")

declare void @"llvm.memcpy.p0i8.p0i8.i32"(i8* %".1", i8* %".2", i32 %".3", i1 %".4")

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

define spir_kernel void @"fill"({i32*, i32} %"buf")
{
entry:
  %"buf.1" = alloca {i32*, i32}
  store {i32*, i32} %"buf", {i32*, i32}* %"buf.1"
  %"spirv_global_id_x" = load i32, i32* @"__spirv_BuiltInGlobalInvocationId_x"
  %"i" = alloca i32
  store i32 %"spirv_global_id_x", i32* %"i"
  store i32 %"spirv_global_id_x", i32* %"i"
  %"i.1" = load i32, i32* %"i"
  %"buf.2" = load {i32*, i32}, {i32*, i32}* %"buf.1"
  %".6" = extractvalue {i32*, i32} %"buf.2", 1
  %"lttmp" = icmp slt i32 %"i.1", %".6"
  br i1 %"lttmp", label %"then", label %"else"
then:
  %"buf.3" = load {i32*, i32}, {i32*, i32}* %"buf.1"
  %".8" = extractvalue {i32*, i32} %"buf.3", 0
  %"p" = alloca i32*
  store i32* %".8", i32** %"p"
  store i32* %".8", i32** %"p"
  %"i.2" = load i32, i32* %"i"
  %"i.3" = load i32, i32* %"i"
  %".11" = load i32*, i32** %"p"
  %".12" = getelementptr i32, i32* %".11", i32 %"i.3"
  store i32 %"i.2", i32* %".12"
  br label %"ifcont"
else:
  br label %"ifcont"
ifcont:
  ret void
}
