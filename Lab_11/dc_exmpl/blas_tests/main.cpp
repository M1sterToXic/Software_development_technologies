#include <iostream>
#include <chrono>
#include <vector>

void mock_blas_operation() {
    std::vector<double> v(1000000, 1.1);
    for(auto& x : v) x *= 2.0;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    mock_blas_operation();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "BLAS_TEST " << diff.count() << std::endl;
    return 0;
}