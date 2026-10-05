; Hand-written IR in the shape rustc emits, names already demangled
; (the test runs the extractor with `cat` as the demangler).
define internal void @"larql_lql::executor::Session::exec_show_models"(ptr %self) {
  %1 = call ptr @"larql_lql::executor::list_dir"()
  ret void
}
define internal void @"larql_lql::executor::Session::exec_hidden"(ptr %self) {
  %1 = call ptr @"larql_lql::executor::Session::peek"(ptr %self)
  %2 = call ptr @"alloc::fmt::format"(ptr %1)
  ret void
}
define internal void @"larql_lql::executor::Session::exec_walk"(ptr %self) {
  %1 = invoke ptr @"larql_lql::executor::Session::require_vindex"(ptr %self) to label %ok unwind label %bad
ok:
  ret void
bad:
  ret void
}
define internal void @"larql_lql::executor::Session::exec_stats::{closure#0}"(ptr %env) {
  %f = load ptr, ptr %env
  %1 = call ptr %f(ptr %env)
  ret void
}
define internal void @"<larql_lql::executor::Session as core::fmt::Debug>::fmt"(ptr %self) {
  %1 = call ptr @"larql_lql::executor::Session::exec_show_models"(ptr %self)
  ret void
}
define internal void @"larql_lql::executor::Session::exec_begin_patch"(ptr %self) {
  ret void
}
define internal void @"larql_lql::executor::Session::exec_stats"(ptr %self) {
  ret void
}
define internal void @"larql_lql::executor::Session::exec_use"(ptr %self) {
  ret void
}
define internal void @"larql_lql::executor::Session::execute"(ptr %self) {
  %1 = call ptr @"larql_lql::executor::Session::exec_hidden"(ptr %self)
  ret void
}
