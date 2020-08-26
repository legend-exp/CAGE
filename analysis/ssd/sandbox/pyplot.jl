using PyCall

function main()
    # testplot()
    testarray()
end

"""
call the matplotlib backend directly with PyCall
"""
function testplot()
    
    mpl = pyimport("matplotlib")
    mpl.use("Qt5Agg") # Qt5Agg, PS (non-graphical)
    plt = pyimport("matplotlib.pyplot")
    x = range(0;stop=2*pi,length=1000); y = sin.(3*x + 4*cos.(2*x));
    plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
    plt.show()
end

"""
play with 2d arrays.  try using IOContext to pretty print
"""
function testarray()

    # method 1 - no type declared
    # arr = reshape([],0,2)
    
    # method 2 - with type
    # arr = Array{Float64}(undef, 2, 2)
    
    # method 3 - from random numbers
    arr = rand(0:.1:1, 100, 200)

    # remember, Julia is ONE-INDEXED
    arr[1,1] = 1

    # this prints everything - probably a bad idea
    # print(arr)
    
    # this is what the REPL is doing (from stackoverflow)
    display(arr)

    
end