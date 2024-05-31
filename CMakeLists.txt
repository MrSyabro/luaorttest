cmake_minimum_required(VERSION 3.5)	 # Проверка версии CMake.
										# Если версия установленой программы
										# старее указаной, произайдёт аварийный выход.

project(luaexe C)			# Название проекта
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

add_subdirectory(luaort)
add_subdirectory(luapng)
add_subdirectory(cutils)

# Попытка найти исполняемый файл
find_program(LUA_EXECUTABLE NAMES lua PATHS /bin /usr/bin /usr/local/bin C:\\lua\\bin ${LUA_BINDIR})

# Проверка наличия исполняемого файла
if(LUA_EXECUTABLE)
    message(STATUS "Исполняемый файл найден: ${LUA_EXECUTABLE}")

    get_filename_component(LUA_DIRECTORY ${LUA_EXECUTABLE} DIRECTORY)
    set(LUA_FOUND TRUE)
    set(LUA_LIBRARIES "${LUA_DIRECTORY}/../lib")
    set(LUA_INCLUDE_DIR "${LUA_DIRECTORY}/../include")

    find_library(LUA_LIBRARY
                NAMES lua54 liblua54 lua liblua
                PATHS ${LUA_DIRECTORY} ${LUA_LIBRARIES})

    link_libraries(${LUA_LIBARRY})
    link_directories(${LUA_DIRECTORY} ${LUA_LIBRARIES})
    include_directories(${LUA_INCLUDE_DIR})
else()
    message(STATUS "Исполняемый файл не найден. Собираю из исходников.")
    
    find_package(Lua REQUIRED)

    # Определение исполняемого файла для сборки
    add_executable(luaexe src/lua.c)
    target_link_libraries(luaexe PUBLIC lua)

    # Настройте установочные команды для библиотек
    install(TARGETS luaexe
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX})

    add_dependencies(luaexe luapng luaort vec)
endif()

file(INSTALL 
        ${CMAKE_CURRENT_SOURCE_DIR}/samples
        DESTINATION ${CMAKE_INSTALL_PREFIX})