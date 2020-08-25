##############################################
# Written by Gulden Othman for CAGE simulation processing. Should work for any g4simple output file
# July, 2020
##############################################

using LegendDataTypes, LegendHDF5IO
using ArraysOfArrays, StaticArrays, Tables, TypedTables
using HDF5
using DataFrames
# using SolidStateDetectors
using Unitful
using Plots; pyplot(fmt = :png);

function main()
    # base_filename = "tracking_allSteps_oppi_ring_y9_norm_241Am_1000000"
    # raw_dir = "../alpha/raw_out/oppi/"
    # processed_dir = "../alpha/processed_out/oppi/"
    raw_extension = ".hdf5"
    processed_extension = ".lh5"

    base_filename = "test_newDet_sourceRotNorm_y6mm_ICPC_Pb_241Am_100000"
    raw_dir = "../alpha/raw_out/"
    processed_dir = "../alpha/processed_out/"
    # println("got to main function")

    processHits_steps(raw_dir, processed_dir, base_filename, raw_extension, processed_extension)

    # processHits_forSSD(raw_dir, processed_dir, base_filename, raw_extension, processed_extension, icpc=true)

end

function processHits_steps(raw_dir, processed_dir, base_filename, raw_extension, processed_extension)
    # Read raw g4simple HDF5 output file and construct arrays of corresponding data
    filename = raw_dir * base_filename * raw_extension
    println("Processing file: $filename")
    g4sfile = h5open(filename, "r")
    g4sntuple = g4sfile["default_ntuples"]["g4sntuple"]

    event = read(g4sntuple["event"]["pages"])
    trackID = read(g4sntuple["trackID"]["pages"])
    parentID = read(g4sntuple["parentID"]["pages"])
    pid = read(g4sntuple["pid"]["pages"])
    step = read(g4sntuple["step"]["pages"])
    KE = read(g4sntuple["KE"]["pages"])
    energy = read(g4sntuple["Edep"]["pages"])
    volID = read(g4sntuple["volID"]["pages"])
    iRep = read(g4sntuple["iRep"]["pages"])

    x = read(g4sntuple["x"]["pages"])
    y = read(g4sntuple["y"]["pages"])
    z = read(g4sntuple["z"]["pages"])

    # Example of how to read data directly from the HDF5 file
    # data = read(y)
    # println("type: ", typeof(data),data)

    # Construct a Julia DataFrame with the arrays we just constructed from the g4sfile data
    raw_df = DataFrame(
        event = event,
        trackID = trackID,
        parentID = parentID,
        pid = pid,
        step = step,
        KE = KE,
        energy = energy,
        volID = volID,
        iRep = iRep,
        x = x,
        y = y,
        z = z
        )

    # Save only events that occur in the detector PV
    gdf = groupby(raw_df, :volID)
    det_hits = DataFrame(gdf[2])

    # Need to turn DataFrame into a table before saving to lh5
    table_df = TypedTables.Table(det_hits)

    out_filename = processed_dir * "joulesmachine_test_processed_" * base_filename * processed_extension

    # println(typeof(out_filename))

    # Save output to .lh5
    h5open(out_filename, "w") do f
        LegendDataTypes.writedata(f, "tracking_detHits", table_df)
    end

    println("Processed file save to: $out_filename")

end

function processHits_forSSD(raw_dir, processed_dir, base_filename, raw_extension, processed_extension; icpc::Bool=false, ppc::Bool=false)
    # Read raw g4simple HDF5 output file and construct arrays of corresponding data
    filename = raw_dir * base_filename * raw_extension
    println("Processing file: $filename")
    g4sfile = h5open(filename, "r")
    g4sntuple = g4sfile["default_ntuples"]["g4sntuple"]

    evtno = read(g4sntuple["event"]["pages"])
    detno = read(g4sntuple["iRep"]["pages"])
    thit = read(g4sntuple["t"]["pages"]).*u"s"
    edep = read(g4sntuple["Edep"]["pages"]).*u"MeV"
    ekin = read(g4sntuple["KE"]["pages"]).*u"MeV"
    volID = read(g4sntuple["volID"]["pages"])
    stp = read(g4sntuple["step"]["pages"])
    mom = read(g4sntuple["parentID"]["pages"])
    trk = read(g4sntuple["trackID"]["pages"])
    pdg = read(g4sntuple["pid"]["pages"])

    x = read(g4sntuple["x"]["pages"])
    y = read(g4sntuple["y"]["pages"])
    z = read(g4sntuple["z"]["pages"])

    # Translate z-coordinates to match SSD geometry
    if icpc==true
        println("Translating z-dimension so surface is at ICPC height")
        z .+= (22.5 + 86.4)
    elseif oppi==true
        println("Translating z-dimension so surface is at ICPC height")
        z .+= (22. + 51.)
    else
        println("Keeping in g4simple coordinates")
    end

    # Construct array of positions for input to SSD
    n_ind = length(evtno)
    # pos = [ (x[i], y[i], z[i]) for i in 1:n_ind]
    pos = [ SVector{3}(([ x[i], y[i], z[i] ] .* u"mm")...) for i in 1:n_ind ]

    # Construct a Julia DataFrame with the arrays we just constructed from the g4sfile data
    raw_df = DataFrame(
            evtno = evtno,
            detno = detno,
            thit = thit,
            edep = edep,
            pos = pos,
            ekin = ekin,
            volID = volID,
            stp = stp,
            mom = mom,
            trk = trk,
            pdg = pdg
        )


    # Save only events that occur in the detector PV
    gdf = groupby(raw_df, :volID)

    # volID = 1 for the detectors in CAGE g4simple sims, this selects that
    det_hits = DataFrame(gdf[2])

    # Need to turn DataFrame into a TypedTable before saving to lh5
    hits = TypedTables.Table(det_hits)

    out_filename = processed_dir * "test_SSD_processed_" * base_filename * processed_extension

    # println(typeof(out_filename))

    # Save output to .lh5
    h5open(out_filename, "w") do f
        LegendDataTypes.writedata(f, "SSD_detHits", hits)
    end

    println("Processed file save to: $out_filename")

end

main()
