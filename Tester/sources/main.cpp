#include <iostream>
#include "Application.hpp"

static void start(const char* pathToFolder) {
    Tester::Application app;
    app.loadDataFromDisk(pathToFolder);
    app.testVertexShader();
}

struct ParsedArguments {
    const char* pathToBinariesFolder = "";
};

ParsedArguments parseCLI(const int argc, char** args) noexcept {
    ParsedArguments arguments;
    if (argc == 2) {
        arguments.pathToBinariesFolder = args[1];
    }
    return arguments;
}

int main(int argc, char** args) {
    try {
        ParsedArguments arguments = parseCLI(argc, args);
        start(arguments.pathToBinariesFolder);
    } catch (const std::exception& e) { std::cerr << "[Error] " << e.what() << std::endl; }

    system("pause");
	return 0;
}