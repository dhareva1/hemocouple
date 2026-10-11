"""Physics-informed neural operators for 1D-3D hemodynamic interface coupling.

Module map (proposal section in brackets):
    geometry, waveforms     parametric cases and pulsatile inflow       [7.2]
    cross_section           quadrature grid on the interface section    [7.3]
    womersley               analytic pulsatile profiles                 [7.2, mock]
    mock                    cheap lumped 1D/"3D" stand-ins for testing  [-]
    dataset                 case records, windows, case-level splits    [7.3, 7.6]
    pod                     POD basis of interface profiles             [7.3]
    model                   POD-DeepONet + exact flux projection        [7.3, 7.4]
    losses, train           objective and training loop                 [7.4]
    providers, coupling     online coupling loop, Mode A / Mode B       [7.5]
    metrics                 errors, mass, reflection, break-even cost   [7.7]
    adapters                interfaces for real solvers (stubs)         [7.2]
"""

__version__ = "0.1.0"
