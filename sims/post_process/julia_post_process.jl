##############################################
# Written by Gulden Othman for CAGE simulation processing. Should work for any g4simple output file in hdf5 format
# July, 2020
##############################################

using LegendDataTypes, LegendHDF5IO
using ArraysOfArrays, StaticArrays, Tables, TypedTables
using HDF5
using DataFrames
using SolidStateDetectors: cluster_detector_hits
using RadiationDetectorSignals: group_by_evtno, ungroup_by_evtno, group_by_evtno_and_detno
using Unitful

function main()

    # specify here the base filename, directories, and file extensions, and choose which processing funciton to run

    base_filename = "test_newDet_sourceRotNorm_y6mm_ICPC_Pb_241Am_100000"
    raw_dir = "../alpha/raw_out/"
    processed_dir = "../alpha/processed_out/"
    raw_extension = ".hdf5"
    processed_extension = ".lh5"

    # processHits_steps(raw_dir, processed_dir, base_filename, raw_extension, processed_extension)

    # processHits(raw_dir, processed_dir, base_filename, raw_extension, processed_extension, icpc=true)

    processEvents_forSSD(raw_dir, processed_dir, base_filename, raw_extension, processed_extension, icpc=true)

end

function processHits_steps(raw_dir, processed_dir, base_filename, raw_extension, processed_extension)

    # Use when you want to process a raw g4simple simulation output (hdf5) with individual particle steps for debugging.
    # Outputs a Table saved into a .lh5 file

    T = Float64
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

    # volID = 1 for the detectors in CAGE g4simple sims, this selects only events in the detector
    det_hits = DataFrame(gdf[2])

    # Need to turn DataFrame into a table before saving to lh5
    table_df = TypedTables.Table(det_hits)

    out_filename = processed_dir * "processed_" * base_filename * processed_extension

    # println(typeof(out_filename))

    # Save output to .lh5
    h5open(out_filename, "w") do f
        LegendDataTypes.writedata(f, "tracking_detHits", table_df)
    end

    println("Processed file save to: $out_filename")

end

function processHits(raw_dir, processed_dir, base_filename, raw_extension, processed_extension; icpc::Bool=false, ppc::Bool=false)
    # Use when you want to process a raw g4simple simulation output (hdf5) into a Table with individual hits separated.
    # (May be useful for debugging, but not readable in SSD)

    # Read raw g4simple HDF5 output file and construct arrays of corresponding data
    filename = raw_dir * base_filename * raw_extension
    println("Processing file: $filename")
    g4sfile = h5open(filename, "r")
    g4sntuple = g4sfile["default_ntuples"]["g4sntuple"]

    evtno = read(g4sntuple["event"]["pages"])
    detno = read(g4sntuple["iRep"]["pages"])
    thit = read(g4sntuple["t"]["pages"]).*u"ns"
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
        println("Translating z-dimension so surface is at OPPI height")
        z .+= (22. + 51.)
    else
        println("Keeping in g4simple coordinates")
    end

    # Construct array of positions for input to SSD
    n_ind = length(evtno)
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

    # volID = 1 for the detectors in CAGE g4simple sims, this selects only events in the detector
    det_hits = DataFrame(gdf[2])

    # Need to turn DataFrame into a TypedTable before saving to lh5
    hits = TypedTables.Table(det_hits)

    out_filename = processed_dir * "test2_SSD_processed_" * base_filename * processed_extension

    # println(typeof(out_filename))

    # Save output to .lh5
    h5open(out_filename, "w") do f
        LegendDataTypes.writedata(f, "detHits", hits)
    end

    println("Processed file save to: $out_filename")

end

function processEvents_forSSD(raw_dir, processed_dir, base_filename, raw_extension, processed_extension; icpc::Bool=false, ppc::Bool=false)
    # Use when you want to process a raw g4simple simulation output (hdf5) into a Table grouped by event
    # This produces a Table in a .lh5 file that can then be directly input into the SSD
    # SolidStateDetectors.simulate_waveforms() function for waveform generation from monte-carlo simulated events


    # Read raw g4simple HDF5 output file and construct arrays of corresponding data
    filename = raw_dir * base_filename * raw_extension
    println("Processing file: $filename")
    g4sfile = h5open(filename, "r")
    g4sntuple = g4sfile["default_ntuples"]["g4sntuple"]

    evtno = read(g4sntuple["event"]["pages"])
    detno = read(g4sntuple["iRep"]["pages"])
    thit = read(g4sntuple["t"]["pages"]).*u"ns"
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

    # Translate z-coordinates to match SSD geometry-- CAGE specific
    if icpc==true
        println("Translating z-dimension so surface is at ICPC height")
        z .+= (22.5 + 86.4)
    elseif oppi==true
        println("Translating z-dimension so surface is at OPPI height")
        z .+= (22. + 51.)
    else
        println("Keeping in g4simple coordinates")
    end

    # Construct array of positions for input to SSD
    n_ind = length(evtno)
    pos = [ SVector{3}(([ x[i], y[i], z[i] ] .* u"mm")...) for i in 1:n_ind ]


    # Construct a Julia DataFrame with the arrays we just constructed from the g4sfile data to make grouping easier
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

    # volID = 1 for the detectors in CAGE g4simple sims, this selects only events in the detector
    det_hits = DataFrame(gdf[2])

    # Need to turn DataFrame into a Table before using internal SSD functions (group_by_evtno, cluster_detector_hits, etc)
    # Only include parameters needed by SSD

    hits_flat = Table(
        evtno = det_hits.evtno,
        detno = det_hits.detno,
        thit = det_hits.thit,
        edep = det_hits.edep,
        pos = det_hits.pos
     )

     # group hits by event number and cluster hits based on distance
    hits_by_evtno = group_by_evtno(hits_flat)
    hits_clustered = cluster_detector_hits(hits_by_evtno, 0.2u"mm")
    hits_by_det = group_by_evtno_and_detno(ungroup_by_evtno(hits_clustered))

    # create output filename and save Table to .lh5
    out_filename = processed_dir * "test2_events_SSD_processed_" * base_filename * processed_extension

    # println(typeof(out_filename))

    # Save output to .lh5
    h5open(out_filename, "w") do f
        LegendDataTypes.writedata(f, "SSD_detEvents", hits_by_det)
    end

    println("Processed file save to: $out_filename")

end

main()
