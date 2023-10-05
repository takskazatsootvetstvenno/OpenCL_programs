#include <iostream>
#include <locale>

#include "Application.hpp"

static void start(const char* pathToFolder) {
    Tester::Application app;
    app.parseTestFolder(pathToFolder);
    app.runTests();
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

void setGlobalLocale() {
    try {
        std::locale::global(std::locale("en_US.UTF-8"));
    } catch (const std::exception& e) {
        std::cerr << "Can't use \"en_US.UTF-8\" locale!" << std::endl;
    }
}

int main(int argc, char** args) {
    setGlobalLocale();
    try {
        ParsedArguments arguments = parseCLI(argc, args);
        arguments.pathToBinariesFolder = "C:/Users/Denis/source/repos/opencl_programs/OpenCL_programs/Tests";
        start(arguments.pathToBinariesFolder);
    } catch (const std::exception& e) { std::cerr << "[Error] " << e.what() << std::endl; }

    system("pause");
	return 0;
}