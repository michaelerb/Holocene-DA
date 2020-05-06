"""

PERFORM A MULTITIMESCALE ENKF-BASED DATA ASSIMILATION RECONSTRUCTION 

Author: Nathan Steiger, LDEO, Columbia University
nsteiger@ldeo.columbia.edu

Tue May  5 18:33:58 EDT 2020

"""

using DataStructures, NCDatasets, LinearAlgebra, Statistics, StatsBase, Dates, JLD2, FileIO


"""

 Load all the variables for the reconstruction

"""
function configRecon()

# Load netdcf data given path
#ds=NCDataset("/Users/nathansteiger/data/formatted_data_Holocene_DA.nc")
ds=NCDataset("/d1/nsteiger/proxy-data/formatted_data_Holocene_DA.nc")


# Format input data -----------------

# Prior/background ensemble
Xb=convert(Array,Transpose(convert(Array,ds["Xb"])))
svl=size(Xb,1) # size of state vector
ensz=size(Xb,2) # number of ensemble members

# Proxy matrix
ym=convert(Array,Transpose(convert(Array,ds["y"])))
nobs=size(ym,1) # number of proxies/obs
reconyrs=size(ym,2) # proxy matrix defines how many years to do the reconstruction for

# Averaging time scale of the proxy data values
ya0=convert(Array,Transpose(convert(Array,ds["ya"])))
# NaNs are not support by an Int array type, so replace with zero, then this 
# conversion to zero won't matter since the NaN screening takes place on the
# proxy data matrix
replace!(ya0, NaN=>0)
ya=convert.(Int,ya0)


# HXb and R are arrays of dictionaries
# define custom array type: array the size of number of proxies containing a 
#  dictionary with keys corresponding to averaging time scale and the ensemble
#  estimate (vector) for each of those time scales 
hxba=Array{Dict{Int64,Array{Float64,1}}}(undef,nobs) 
ra=Array{Dict{Int64,Float64}}(undef,nobs) 

# Loop through proxies and nested groups to load HXb and R
rn=Array{Int64}(undef,nobs)
for i in 1:nobs
   tmres=keys(ds.group["$(i)"].group) # get time resolutions
   # Make dictionaries with all the data using an internal iterator
   hxba[i]=Dict(parse(Int,j)=>ds.group["$(i)"].group[j]["HXb"][:] for j in tmres)
   ra[i]=Dict(parse(Int,j)=>ds.group["$(i)"].group[j]["R"][1] for j in tmres)

   # find which R values are NaNs (hack around lack of pre-screening)
   rn[i]=sum(isnan.(collect(values(ra[i]))))
end

# Remove proxies with NaN R values
ridx=findall(f->f>0,rn)
deleteat!(ra,ridx)
deleteat!(hxba,ridx)
kidx=findall(iszero,rn)
nobs=size(kidx,1)
ym=ym[kidx,:]
ya=ya[kidx,:]

return (Xb,ym,ya,hxba,ra,svl,nobs,reconyrs,ensz)

end



"""

Compute a multitime-scale DA reconstruction

"""

function msrecon(Xb::Array{Float64,2}, ym::Array{Float64,2}, ya::Array{Int,2}, hxba::Array{Dict{Int64,Array{Float64,1}}}, ra::Array{Dict{Int64,Float64}}, svl::Int, nobs::Int, reconyrs::Int, ensz::Int)

# Random sample of ensemble to save
rens=sample(collect(1:ensz),100;replace=false,ordered=false)
Xrens=Array{Float64}(undef,svl,size(rens,1),reconyrs) # only size of subsample
# Save full ensemble
#Xrens=Array{Float64}(undef,svl,ensz,reconyrs) # full size

# Ensemble mean
Xrm=Array{Float64}(undef,svl,reconyrs)

# LOOP OVER TIME STEPS
#Threads.@threads for t in 1:reconyrs # can try parallelizing the loops
for t in 1:reconyrs

   # Find which values are not NaNs
   ns=findall(!isnan,ym[:,t])

   # Do the reconstruction as long as there are proxies to assimilate
   if size(ns,1)>0

      # Extract proxy values that aren't NaNs
      y=ym[ns,t]

      # Extract HXb and R of correct proxy and averaging time scale 
      HXb=Array{Float64}(undef,size(ns,1),ensz)
      r0=Array{Float64}(undef,size(ns,1))
      j=1
      for i in ns
         HXb[j,:]=hxba[i][ya[i,t]]
	 r0[j]=ra[i][ya[i,t]]
         j+=1
      end
      # R values in diagonal matrix (assumes uncorrelated errors)
      R=Diagonal(r0)

      # Assimilate proxies
      (Xa,Xam)=damup(Xb,HXb,R,y)
   else
      # No proxies are assimilated
      Xa=Xb
      Xam=mean(Xb,dims=2)
   end

   # Save analysis mean and sample of ensemble for each time step
   Xrm[:,t]=Xam
   Xrens[:,:,t]=Xa[:,rens] # size of subsample
   #Xrens[:,:,t]=Xa # full size

   # Note progression of the reconstruction
   if mod(t,100)==0
      println("Time step $(t) complete...")
   end

end


return (Xrm, Xrens)

end



"""

Data assimilation matrix update step, assimilating all observations
for a given time step at once. Variables with their dimensions are 
indicated by [dim1 dim2] given below. This set of update equations
follow those from Whitaker and Hamill 2002: Eq. 2, 4, 10, & Sec 3.

ARGUMENTS:
    Xb = background (prior) [statevector,ens]
    y = observation (with implied noise) [vector]
    HXb = model estimate of observations H(Xb) [size(y) ens]
    R = diagonal observation error variance matrix [size(y) size(y)]
    infl = inflation factor [scalar] **Note: modify code to include**

RETURNS:
    Xa = analysis (posterior) [statevector ens]
    Xam = analysis mean [statevector]

"""

function damup(Xb::Array{Float64,2}, HXb::Array{Float64,2}, R::Diagonal{Float64,Array{Float64,1}}, y::Array{Float64,1})

# Ensemble size for decompositions and means
nens=size(Xb,2)

# Decompose Xb and HXb into mean and perturbations (for Eqs. 4 & Sec 3)
Xbm=mean(Xb,dims=2)
Xbp=Xb-repeat(Xbm,1,nens)

HXbm=mean(HXb,dims=2)
HXbp=HXb-repeat(HXbm,1,nens)

# Apply inflation if chosen to be != 1
#if infl[1] != 1
#   Xbp=infl*Xbp
#end

# Kalman gain for mean and matrix covariances (Eq. 2)
PbHT=(Xbp*HXbp')./(nens-1)
HPbHTR=(HXbp*HXbp')./(nens-1)+R
K=PbHT*inv(HPbHTR)

# Kalman gain for the perturbations (Eq. 10)
sHPbHTR=sqrt(HPbHTR)
sR=sqrt(R)
Ktn=PbHT*(inv(sHPbHTR))'
Ktd=inv(sHPbHTR+sR)
Kt=Ktn*Ktd

# Update mean and perturbations (Eq. 4 & Sec 3)
Xam=Xbm+K*(y-HXbm)
Xap=Xbp-Kt*HXbp

# Reconstitute the full ensemble state vector
Xa=Xap+repeat(Xam,1,nens)

# Output both the full ensemble and the ensemble mean
return (Xa, Xam)
#return (Xa) # return only full ensemble

end




"""

Perform the reconstruction

"""


# Load the input data ------------------------------
println("Loading input data...")
# Run the configuration and loading of variables
(Xb,ym,ya,hxba,ra,svl,nobs,reconyrs,ensz)=configRecon()


# Run the reconstruction ---------------------------
println("Computing the reconstruction...")
@time (Xrm,Xrens)=msrecon(Xb,ym,ya,hxba,ra,svl,nobs,reconyrs,ensz)


# Save the output ----------------------------------

#vrnm="holocene_recon_$(now()).jld2"
vrnm="holocene_recon_$(now()).nc"
println("Saving the reconstruction as "*vrnm)

# JLD2 format
#@save "/d2/nsteiger/holocene-da-output/"*vrnm Xrm Xrens

# NetCDF format
ds = Dataset("/d2/nsteiger/holocene-da-output/"*vrnm,"c")
defDim(ds,"state",svl)
defDim(ds,"recontimes",reconyrs)
defDim(ds,"ens",size(Xrens,2))
ds.attrib["title"] = "Holocene climate reconstruction"
Xm = defVar(ds,"Xm",Float32,("state","recontimes"), attrib = OrderedDict("units" => "Degree Kelvin"))
Xens = defVar(ds,"Xens",Float32,("state","ens","recontimes"), attrib = OrderedDict("units" => "Degree Kelvin"))
Xm[:,:]=Xrm
Xens[:,:,:]=Xrens
close(ds)


println("Reconstruction complete")



