#include <iostream>
#include "Application.hpp"

static void start() {
    Tester::Application app;
    app.test();
}

int main()
{
    try {
        start();
    } catch (const std::exception& e) { std::cerr << "[Error] " << e.what() << std::endl; }

    system("pause");
	return 0;
}