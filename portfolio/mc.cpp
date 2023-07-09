/*
env /usr/bin/arch -x86_64 /bin/zsh --login
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) mc.cpp -o mcpp$(python3-config --extension-suffix) -undefined dynamic_lookup
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void genPaths(py::array_t<double> noise, 
              py::array_t<double> omega,
              py::array_t<double> alpha,
              py::array_t<double> beta, 
              py::array_t<double> sigma_last,
              py::array_t<double> result) 
{
    assert(noise.ndim() == 3);
    assert(result.ndim() == 3);
    assert(sigma_last.ndim() == 3);

    auto n = noise.mutable_unchecked<3>();
    auto r = result.mutable_unchecked<3>();
    auto s = sigma_last.mutable_unchecked<3>();
    for (size_t dim = 0; dim < 3; dim++) {
        assert(n.shape(dim) == r.shape(dim));
        assert(n.shape(dim) == s.shape(dim));
    }

    auto o = omega.mutable_unchecked<1>();
    auto a = alpha.mutable_unchecked<1>();
    auto b = beta.mutable_unchecked<1>();

    for (py::ssize_t path = 0; path < r.shape(0); path++) {
        for (py::size_t step = 0; step < r.shape(1); step++) {
            for (py::size_t asset = 0; asset < r.shape(2); asset++) {
                if (step > 0) {
                    // sigma^2 = omega + alpha * dS^2 + beta * sigma^2
                    s(path, step, asset) = o(asset) + a(asset) * pow(r(path, step-1, asset), 2.0) + b(asset) * s(path, step-1, asset);
                }
                // dS = sigma * noise
                r(path, step, asset) = sqrt(s(path, step, asset)) * n(path, step, asset);
            }
        }
    }
}

PYBIND11_MODULE(mcpp, m) {
    m.doc() = "fast mc path generation";
    m.def("genPaths", &genPaths, "generate MC paths");
}
