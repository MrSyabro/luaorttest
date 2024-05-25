cmake_minimum_required(VERSION 3.5)	 # Проверка версии CMake.
										# Если версия установленой программы
										# старее указаной, произайдёт аварийный выход.

project(luaexe C)			# Название проекта
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

find_package(Lua REQUIRED)

add_executable(luaexe src/lua.c)
target_link_libraries(luaexe PUBLIC lua)

add_subdirectory(luaort)
add_subdirectory(luapng)

# Настройте установочные команды для библиотек
install(TARGETS luaexe
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX})

add_dependencies(luaexe luapng luaort)

file(INSTALL 
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test.lua
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/candy.onnx
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/in.png
        DESTINATION ${CMAKE_INSTALL_PREFIX})

file(INSTALL ${ORT_DIR}/onnxruntime.dll DESTINATION ${CMAKE_INSTALL_PREFIX})
file(INSTALL ${ORT_DIR}/onnxruntime_providers_shared.dll DESTINATION ${CMAKE_INSTALL_PREFIX})