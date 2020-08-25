using Plots; pyplot(show=true) 
using SolidStateDetectors

"""
clint's oppi simulation suite with SolidStateDetectors
"""
function main()
    sim = simulate()
end


function simulate(show=false)
    sim = Simulation{Float32}("ivc.json")
    if show
        plot(sim.detector)
        plot!(size=(800,600)) # the window size and position is finicky
    end
    return sim
end
