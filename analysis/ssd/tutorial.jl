# lukas says to always load Plots and set backend first 
using Plots; pyplot(show=true) # need this when running from terminal
using Unitful
using SolidStateDetectors # alias: SSD, sometimes?

using HDF5
using LegendHDF5IO: readdata, writedata
# using LegendDataTypes
# using LegendTextIO

"""
SolidStateDetetctors tutorial adapted from lukas:
https://github.com/lmh91/legend-julia-tutorial
"""
function main()
    sim = config()
    # initial(sim)
    # epotential(sim)
    # partial_depletion(sim)
    # efield(sim)
    # drift(sim)
    # wpot(sim)
    # io(sim)
    load_h5()
    # quick()
end


function config(show=false)
    sim = Simulation{Float32}("ivc.json")
    SSD.apply_initial_state!(sim) # efield calc errors without this
    if show
        plot(sim.detector, size=(800,600))
    end
    return sim
end
    
    
function initial(sim, show=false)
    SSD.apply_initial_state!(sim)
    
    if show
        plot(
            plot(sim.electric_potential),
            plot(sim.point_types),
            plot(sim.ρ), # charge density distribution
            plot(sim.ϵ), # dielectric distribution
            layout = (1, 4), size = (1400, 700)
        )
    end
end


function epotential(sim, show=false)
    
    # would be good to be able to save the output of this to a file
    # s/t we don't have to keep re-running this.  maybe the IO section
    # below does this.
    SSD.calculate_electric_potential!(sim, max_refinements = 4)
    
    if show
        plot(
            plot(sim.electric_potential, φ = 20),
            plot(sim.point_types), 
            plot(sim.ρ), 
            plot(sim.ϵ), 
            layout = (1, 4), size = (1400, 700)
        )
    end
    
    get_active_volume(sim.point_types)
end


function partial_depletion(sim)
    det_undep = deepcopy(sim.detector)
    det_undep.contacts[end].potential = 200; # bias voltage (V)
    sim_undep = Simulation(det_undep);
    
    SSD.apply_initial_state!(sim)
    SSD.apply_initial_state!(sim_undep)
    
    SSD.calculate_electric_potential!(sim_undep, 
        depletion_handling = true,     
        convergence_limit=1e-6, 
        max_refinements = 4, 
        verbose = false
    );
    
    plot(
        plot(sim_undep.electric_potential), 
        plot(sim_undep.point_types), 
        layout = (1, 2), size = (800, 700)
    )
    
    # note: this function requires apply_initial_state to be run previously
    print("Original active vol: ", get_active_volume(sim.point_types))
    print("Undeplet active vol: ", get_active_volume(sim_undep.point_types));
end


function efield(sim, show=false)
    SSD.apply_initial_state!(sim)
    SSD.calculate_electric_potential!(sim, max_refinements = 4)
    SSD.calculate_electric_field!(sim, n_points_in_φ = 72)
    if show
        plot_electric_field(sim, φ = 0)
    end
    return SSD
end

"""
requires Unitful
"""
function drift(sim, epot, efield)
    T = Float32
    SSD.calculate_electric_potential!(sim, max_refinements = 4)
    SSD.calculate_electric_field!(sim, n_points_in_φ = 72)
    
    drift_model = ADLChargeDriftModel();
    SSD.set_charge_drift_model!(sim, drift_model)
    SSD.apply_charge_drift_model!(sim)
    
    starting_positions = [
        CylindricalPoint{T}( 0.02, deg2rad(10), 0.015 ), 
        CylindricalPoint{T}( 0.015, deg2rad(20), 0.045 ), 
        CylindricalPoint{T}( 0.025, deg2rad(30), 0.025 )
    ]
    
    energy_depos = T[1460, 609, 1000] * u"keV" # needed in signal generation
    
    event = SSD.Event(starting_positions, energy_depos);
    
    time_step = 5u"ns"
    SSD.drift_charges!(event, sim, Δt = time_step)
    
    plot(sim.detector, size = (700, 700))
    plot!(event.drift_paths)
end


function wpot(sim)

    # doesn't require these to be run already
    # SSD.apply_initial_state!(sim)
    # SSD.calculate_electric_potential!(sim, max_refinements = 4)
    # SSD.calculate_electric_field!(sim, n_points_in_φ = 72)
    
    for contact in sim.detector.contacts
        SSD.calculate_weighting_potential!(
            sim, contact.id, 
            max_refinements = 4, 
            n_points_in_φ = 2, 
            verbose = false
        )
    end    
    plot(  
        plot(sim.weighting_potentials[1]),
        plot(sim.weighting_potentials[2]),
        size = (900, 700)
    )
end

"""
requires HDF5, LegendHDF5IO
"""
function io(sim)
    
    # show sim status
    print(sim)
    
    # these don't seem to be needed for the HDF5 output to work
    # SSD.apply_initial_state!(sim)
    # SSD.calculate_electric_potential!(sim, max_refinements = 4)
    
    SSD.calculate_electric_field!(sim, n_points_in_φ = 72)
    
    for contact in sim.detector.contacts
        calculate_weighting_potential!(
            sim, contact.id, 
            max_refinements = 4, 
            n_points_in_φ = 2, 
            verbose = false
        )
    end   
    
    drift_model = ADLChargeDriftModel();
    SSD.set_charge_drift_model!(sim, drift_model)
    apply_charge_drift_model!(sim) 
    
    # show updated status
    print("\n", sim)

    # write to HDF5 file using LegendHDF5IO and NamedTuple
    fout = "data/tutorial.h5"
    if !ispath(dirname(fout)) 
        mkpath(dirname(fout)) 
    end
    HDF5.h5open(fout, "w") do h5f
        writedata( h5f, "tutorial", NamedTuple(sim)  )
    end
    
    # load hdf5 file and verify we get all the data back
    sim_new = HDF5.h5open(fout, "r") do h5f
        namtup = readdata(h5f, "tutorial") 
        Simulation(namtup);
    end
    SSD.set_charge_drift_model!(sim_new, ADLChargeDriftModel())
    
    # print status again
    print("\n", sim_new)
end


function load_h5()
    fin = "data/tutorial.h5"
    sim_new = HDF5.h5open(fin, "r") do h5f
        namtup = readdata(h5f, "tutorial") 
        Simulation(namtup);
    end
    SSD.set_charge_drift_model!(sim_new, ADLChargeDriftModel())
    print("\n", sim_new)
end


function quick()
    qsim = Simulation("ivc.json")
    simulate!(qsim);
    plot( 
        plot(qsim.electric_potential),
        plot(qsim.weighting_potentials[1]),
        plot(qsim.weighting_potentials[2]),
        size = (1200, 500), layout = (1, 3)
    )
end


# function wfs()
# 
# 
# end