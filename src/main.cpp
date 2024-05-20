#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <matplot/matplot.h>
#include <cmath>
//#include <AudioFile.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

std::vector <double> sine(double freq , double amp, double length, int sample_rate, double offset) 
{
    int samples = static_cast<int>(length * sample_rate);
    std::vector<double> sine(samples);
    for (int i = 0; i < samples; i++) 
    {
        sine[i] = amp * std::sin(2 * M_PI * freq  * i/sample_rate) + offset;
    }
    return sine;
}

std::vector <double> cosine(double freq , double amp, double length, int sample_rate, double offset) 
{
    int samples = static_cast<int>(length * sample_rate);
    std::vector<double> cosine(samples);
    for (int i = 0; i < samples; i++) 
    {
        cosine[i] = amp * std::cos(2 * M_PI * freq  * i/sample_rate) + offset;
    }
    return cosine;
}

std::vector <double> square(double freq , double amp, double length, int sample_rate, double offset, double duty_cycle) 
{
    int samples = static_cast<int>(length * sample_rate);
    std::vector<double> square(samples);
    double period = 1.0 / freq ;
    duty_cycle=duty_cycle/100.0;
    for (int i = 0; i < samples; i++) 
    {
        square[i] = ((std::fmod(i / sample_rate, period) < (period * duty_cycle)) ? amp : -amp) + offset;
    }
    return square;
}

std::vector <double> triangle(double freq , double amp, double length, int sample_rate, double offset)
{
    int samples = static_cast<int>(length * sample_rate);
    std::vector<double> triangle(samples);
    double period = 1.0 / freq ;
    for (int i = 0; i < samples; i++) 
    {
        double cycle_position = (std::fmod((i / sample_rate), period) / period);
        triangle[i] = (cycle_position < 0.5) ? (2.0 * amp * cycle_position) + offset : (2.0 * amp * (1.0 - cycle_position)) + offset;
    }
    return triangle;
}

std::vector <double> sawtooth(double freq, double amp, double length, int sample_rate, double offset) 
{
    int samples = static_cast<int>(length * sample_rate);
    std::vector<double> sawtooth(samples);
    double period = 1.0 / freq;
    for (int i = 0; i < samples; i++) 
    {
        sawtooth[i] = ((2.0 * amp / period) * ((i/sample_rate) - period * std::floor((i/sample_rate)/ period + 0.5))) + offset;
    }
    return sawtooth;
}

std::pair<std::vector<double>, std::vector<double>> dft(const std::vector<double>& signal)
{
    int n = signal.size();
    std::vector<double> real(n), imag(n);
    for (int k = 0; k < n; k++) {
        real[k] = 0;
        imag[k] = 0;
        for (int i = 0; i < n; i++) {
            double angle = 2 * M_PI * k * i / n;
            real[k] += signal[i] * std::cos(angle);
            imag[k] -= signal[i] * std::sin(angle);
        }
    }

    return {real, imag};
}

std::vector<double> idft(const std::vector<double>& real, const std::vector<double>& imag) 
{
    int n = real.size();
    std::vector<double> signal(n);
    for (int i = 0; i < n; i++) 
    {
        signal[i] = 0;
        for (int k = 0; k < n; k++) 
        {
            double angle = 2 * M_PI * k * i / n;
            signal[i] += real[k] * std::cos(angle) - imag[k] * std::sin(angle);
        }
        signal[i] /= n;
    }

    return signal;
}

void peak(std::vector <double> &signal, double length, int sample_rate)
{
    int samples = static_cast<int>(length * sample_rate);
    std::vector<double> time(samples);
    for (int i = 0; i < samples; i++) 
    {
        time[i] = i / sample_rate;
    }
    std::vector<int> peaks;
    for (int i = 1; i < signal.size()-1; i++) 
    {
        if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
            peaks.push_back(i);
        }
    }
    std::vector<double> m_time;
    std::vector<double> m_signal;
    for(int i=0; i<samples; i++)
    {
        for(int j=0; j<peaks.size(); j++)
        {
            if(peaks[j]==i)
            {
                m_time.push_back(time[i]);
                m_signal.push_back(signal[i]);
            }
        }   
    }
    using namespace matplot;
    auto fig = figure(true);
    auto ax = fig->add_axes();
    ax->plot(time, signal)->line_width(1).color("b");
    hold(on);
    ax->plot(m_time, m_signal, "or");
    ax->xlabel("Time (s)");
    ax->ylabel("Amplitude");
    show();
}

void plot_dft(const std::vector<double>& real, const std::vector<double>& imag, int sample_rate) 
{
    int n = real.size();
    std::vector<double> magnitude(n);
    std::vector<double> frequencies(n);

    for (int k = 0; k < n; k++) 
    {
        magnitude[k] = std::sqrt(real[k] * real[k] + imag[k] * imag[k]);
        frequencies[k] = k * sample_rate / n;
    }
    using namespace matplot;
    auto fig = figure(true);
    auto ax = fig->add_axes();
    ax->plot(frequencies, magnitude);
    ax->xlabel("Frequency (Hz)");
    ax->ylabel("Magnitude");
    ax->title("DFT Magnitude Spectrum");
    ax->grid(true);
    show();
}

void plot(std::vector<double> &signal, double length, double sample_rate) 
{
    int samples = static_cast<int>(length * sample_rate);
    std::vector<double> time(samples);
    for (int i = 0; i < samples; i++) 
    {
        time[i] = i / sample_rate;
    }
    using namespace matplot;
    auto fig = figure(true);
    auto ax = fig->add_axes();
    ax->plot(time, signal);
    ax->xlabel("Time (s)");
    ax->ylabel("Amplitude");
    show();
}

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           sine
           sine_wave
           plot
    )pbdoc";

    m.def("sine", &sine, R"pbdoc(Generates sinus)pbdoc");

    m.def("square", &square, R"pbdoc(Generates pulse wave)pbdoc");

    m.def("triangle", &triangle, R"pbdoc(Generates triangle wave)pbdoc");

    m.def("sawtooth", &sawtooth, R"pbdoc(Generates sawtooth wave)pbdoc");

    m.def("cosine", &cosine, R"pbdoc(Generates cosinus)pbdoc");

    m.def("plot", &plot, R"pbdoc(Plots function)pbdoc");

    m.def("dft", &dft), R"pbdoc(Computes DFT)pbdoc";

    m.def("plot_dft", &plot_dft), R"pbdoc(Plots DFT)pbdoc";

    m.def("idft", &idft, R"pbdoc(Computes IDFT)pbdoc");

    m.def("peak", &peak, R"pbdoc(Searches for peaks in signal and plots this signal with peaks marked)pbdoc");
    
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
