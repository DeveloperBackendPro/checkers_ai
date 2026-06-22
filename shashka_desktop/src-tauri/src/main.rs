// Windows'da release build'da konsol oynasi chiqmasligi uchun

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
fn main() {
    shashka_desktop_lib::run();
}