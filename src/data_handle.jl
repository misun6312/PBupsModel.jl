## Data Import
# get trial data
function TrialData(rawdata, trial::Int)
    if rawdata["pokedR"][trial] > 0
        rat_choice = 1;  # "R" 
    else
        rat_choice = -1; # "L"
    end;

    if typeof(rawdata["rightbups"][trial]) <: Array
        rvec = vec(rawdata["rightbups"][trial])::Array{Float64,1};
    else
        rvec = Float64[rawdata["rightbups"][trial]] 
    end
    if typeof(rawdata["leftbups"][trial]) <: Array
        lvec = vec(rawdata["leftbups"][trial])::Array{Float64,1};
    else
        lvec = Float64[rawdata["leftbups"][trial]] 
    end

    return rvec, lvec,
    rawdata["T"][trial]::Float64, rat_choice
end
