using Combinatorics  # For combinations
using WignerSymbols  # For wigner3j symbols, https://github.com/Jutho/WignerSymbols.jl
using SparseArrays   # For sparse arrays

############################################## Bit functions ##############################################

# From Hacker's Delight, p.66: Counts # 1s in base-10 number x.
# Counts the number of 1s in state/number x (physically, the number of occupied sites in the state x). 

function countBits(x::Int)               
    x = x - ((x >> 1) & 0x55555555)    
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003F
end

function countSignedBits(x::Int)
    # Start with zero and then one, etc.
    odd_bits  = x & 0xAAAAAAAA  # select bits in odd positions
    even_bits = x & 0x55555555  # select bits in even positions
    signVal = countBits(odd_bits) - countBits(even_bits)
    return signVal
end


########################################## Hamiltonian structures ##########################################

# Defining structures for the Hamiltonian to pack all the information
# I can use this for both basis_number and basis_number_mz

# These structures are for the fuzzy sphere using pseudopotentials: YC's way

struct HIsing
    N::Int 
    half_L::Int
    h::Float64
    v00::Float64
    v11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsing(;N=2,half_L=2,h=0,v00=0,v11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,h,v00,v11,stateList,basisMap)
end


struct HIsingZ2
    N::Int 
    half_L::Int
    parityZ2::Int
    h::Float64
    v00::Float64
    v11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsingZ2(;N=2,half_L=2,parityZ2=1,h=0,v00=0,v11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,parityZ2,h,v00,v11,stateList,basisMap)
end


struct HIsingZ2Rot
    N::Int 
    half_L::Int
    parityZ2::Int
    rot::Int
    h::Float64
    v00::Float64
    v11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsingZ2Rot(;N=2,half_L=2,parityZ2=1,rot=1,h=0,v00=0,v11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,parityZ2,rot,h,v00,v11,stateList,basisMap)
end

# These structures are for the alternative way using (g_0,g_1) of the U_l potential instead of the pseudopotentials

struct HIsingAlt
    N::Int 
    half_L::Int
    h::Float64
    g00::Float64
    g11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsingAlt(;N=2,half_L=2,h=0,g00=0,g11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,h,g00,g11,stateList,basisMap)
end


struct HIsingZ2Alt
    N::Int 
    half_L::Int
    parityZ2::Int
    h::Float64
    g00::Float64
    g11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsingZ2Alt(;N=2,half_L=2,parityZ2=1,h=0,g00=0,g11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,parityZ2,h,g00,g11,stateList,basisMap)
end


struct HIsingZ2RotAlt
    N::Int 
    half_L::Int
    parityZ2::Int
    rot::Int
    h::Float64
    g00::Float64
    g11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsingZ2RotAlt(;N=2,half_L=2,parityZ2=1,rot=1,h=0,g00=0,g11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,parityZ2,rot,h,g00,g11,stateList,basisMap)
end

# These structures are for the spherical harmonics regularization

struct HNewIsing
    N::Int 
    half_L::Int
    radius::Float64
    h::Float64
    g00::Float64
    g11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HNewIsing(;N=2,half_L=2,radius=0,h=0,g00=0,g11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,radius,h,g00,g11,stateList,basisMap)
end

struct HNewIsingZ2
    N::Int 
    half_L::Int
    radius::Float64
    parityZ2::Int
    h::Float64
    g00::Float64
    g11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HNewIsingZ2(;N=2,half_L=2,radius=0,parityZ2=1,h=0,g00=0,g11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,radius,parityZ2,h,g00,g11,stateList,basisMap)
end

struct HNewIsingZ2Rot
    N::Int 
    half_L::Int
    radius::Float64
    parityZ2::Int
    rot::Int
    h::Float64
    g00::Float64
    g11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HNewIsingZ2Rot(;N=2,half_L=2,radius=0,parityZ2=1,rot=1,h=0,g00=0,g11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,radius,parityZ2,rot,h,g00,g11,stateList,basisMap)
end


############################################# Basis functions #############################################

# Makes all possible combinations of placing 'N' 1s in a list of length 'L' (twice the physical length)
# Combinations(1:L, N) creates a position list where the 1s are placed. Ex: L=3, N=2: [1,2] [1,3] [2,3]

function basis_number(N::Int, L::Int)
    @assert (0<=N<=L) "Filling outside range"
    numbers = Int[]
    for indices in combinations(1:L, N)
        num = 0                           # Then, the idea is to put 1s in the values [i,j] in the number. 
        for index in indices              # Set the bit at the specified index to be equal to 1
            num |= 1 << (L - index)       # Takes empty array 0 and puts the 1 in the position of the index 
        end
        push!(numbers, num)
    end
    return numbers
end

######################### VERY IMPORTANT MZ IS TWICE THE VALUE OF MAGNETIZATION M #########################

function mz_max_value(N::Int, half_L::Int)
    if (mod(N,2) == 0)
        mz_max = N*(half_L-div(N,2))
    elseif (mod(N,2) == 1)
        mz_max = N*(half_L-1) - div((N-1)^2,2)
    end
    return mz_max
end


function basis_number_mz(N::Int, half_L::Int, mz::Int=0)
    mz_max = mz_max_value(N,half_L)
    # The mz_max above is twice the max magnetization for a given filling N
    @assert (mz_max >= 0) & (-mz_max <= mz <= mz_max) "Magnetization value outside range"
    basisNumberMz = Int[]
    numbers_list = basis_number(N, 2*half_L)
    s = half_L - 1
    spacing = 2
    
    for number in numbers_list                 
        counting = 0
        i = 0
        M = 0
        auxnumber = number
        while (counting < N)
            state = number & ((1 << spacing) - 1)
            number = number >> spacing
            
            totBits = countBits(state)
            M += (-s+2*i)*totBits

            counting += totBits
            i += 1
        end 
        if M == mz
            push!(basisNumberMz, auxnumber)   
        end
    end
    return basisNumberMz
end


# Gives list of numbers with filling N and length L and the basisMap mapping those numbers to a basis.

function makeBasisMap(N::Int,L::Int)
    basisMap = Dict()
    stateID = 0
    stateList = basis_number(N, L)
    for state in stateList
        stateID += 1
        basisMap[state] = stateID
    end
    return (stateList,basisMap)
end

# Same as before but for a given magnetization M where Mz= 2M.
# Remember that M is defined as M = \sum_{m,\sigma} m (c^\dagger_{m,\sigma} c_{m,\sigma})

function makeBasisMapMz(N::Int,half_L::Int,mz::Int=0)
    basisMap = Dict()
    stateID = 0
    stateList = basis_number_mz(N,half_L,mz)
    for state in stateList
        stateID += 1
        basisMap[state] = stateID
    end
    return (stateList,basisMap)
end

# This implements the Z2 Ising symmetry of a number with a phase

function swapBits(n::Int)
    # Mask and shift the odd and even bits:
    odd_bits  = n & 0xAAAAAAAA  # select bits in odd positions
    even_bits = n & 0x55555555  # select bits in even positions
    shifted_odd = odd_bits >> 1
    new_number = (shifted_odd) | (even_bits << 1)
    phase = countBits(shifted_odd & even_bits)
    return (new_number, (-1)^phase)
end

# This is fine, it is just giving the representative I will use. There's no need to track the phase (only for the anti/symmetric states)
# This gives the Z2 basis for some parity z2

function basisZ2(stateList, z2::Int)
    basisMap = Dict()
    stateID = 0 
    basisZ2 = Int[]

    for state in stateList
        (state_new, factor) = swapBits(state)
        if state < state_new
            stateID += 1
            basisMap[state] = stateID
            push!(basisZ2, state)
        end
        if state == state_new
            if factor == z2
                stateID += 1
                basisMap[state] = stateID
                push!(basisZ2, state)
            end
        end
    end
    return (basisZ2, basisMap)
end

# This checks whether the state is in or outside the symmetry sector. It returns the state, factor, nb.

function stateZ2(state, parityZ2::Int)

    (state_new, factor) = swapBits(state)

    ### Three options: either the state_new > state to which we don't do anything just return state, or state_new < state to which we need to 
    ### trade the state with |state> -> factor*X |state_new>. But, afterwards, we need to apply the P=(1+ parity*X) meaning that the output is
    ### P|state> -> P(factor*X|state_new>) = (factor*parity) P|state_new>. In both these cases, the number of states in the representative is nb=2.
    ### Finally, the case where state = state_new. Here, nb=1 and it depends what's the parity of the state. If factor = parity, then we're still
    ### in the same symmetry sector and we return just the same state = state_new. If factor != parity, then we are outside the symmetry sector and
    ### so we should get zero. 

    if state < state_new 
        nb = 2
        return (state, 1, nb)
    elseif state > state_new 
        nb = 2
        return (state_new, parityZ2*factor, nb)
    elseif state == state_new
        nb = 1
        if factor == parityZ2
            return (state, 1, nb)
        else 
            return (state, 0, nb)
        end
    end 
end


## For the pi rotation around the y direction:

function reflectionBits(state::Int, half_L::Int, N::Int)
    # Split the number into two: upper_half and lower_half. Upper part is reflected in bits and then in the lower_half of the new state.
    # The lower half is reflected in bits and then in the upper_half of the new state.
    # If odd there's a orbital that doesn't move the mz = 0
    
    if (mod(half_L,2) == 0)
        state_upper = state >> half_L
        state_lower = state & ((1 << half_L) - 1)

        s = half_L-1 # May be unnecesary
        spacing = 2
    
        i = 1
        phase = 0
        counting = 0
        outstate = 0

        bits_up = countBits(state_upper)
        bits_down = countBits(state_lower)

        if ((mod(bits_up,2)==1) && (mod(bits_down,2)==1))
            phase +=1
        end
    
        while (counting < N)
            number_upper = state_upper & ((1 << spacing) - 1)        # This gets the last two numbers
            state_upper = state_upper >> spacing                     # And deletes it from the previous number
            bits_number_upper = countBits(number_upper)              # Bits in the two bits
            bits_state_upper = countBits(state_upper)                # Bits in the rest of the upper number

            if ((mod(bits_number_upper,2) == 1) && (mod(bits_state_upper,2)==1) )
                phase += 1
            end

            number_lower = state_lower & ((1 << spacing) - 1)        # This gets the last two numbers
            state_lower = state_lower >> spacing                     # And deletes it from the previous number
            bits_number_lower = countBits(number_lower)              # Bits in the two bits
            bits_state_lower = countBits(state_lower)                # Bits in the rest of the lower number 

            if ((mod(bits_number_lower,2) == 1) && (mod(bits_state_lower,2)==1) )
                phase += 1
            end

            counting += bits_number_upper + bits_number_lower

            outstate += number_lower*2^(half_L+half_L-2*i)+number_upper*2^(half_L-2*i)

            i+=1 
        end
        return (outstate,(-1)^phase)
        
    else
        spacing = 2
        i = 1
        phase = 0
        counting = 0
        outstate = 0
        
        state_upper = state >> (half_L+1)
        state_lower = state & ((1 << (half_L-1)) - 1)

        state_aux = state >> (half_L-1)
        state_middle = state_aux & ((1 << spacing) - 1)

        bits_up = countBits(state_upper)
        bits_down = countBits(state_lower)
        bits_middle = countBits(state_middle)

        if ((mod(bits_up,2)==1) && (mod(bits_middle,2)==1))
            phase +=1
        end
        
        if ((mod(bits_down,2)==1) && (mod(bits_middle,2)==1))
            phase +=1
        end

        if ((mod(bits_up,2)==1) && (mod(bits_down,2)==1))
            phase +=1
        end

        N_aux = N - bits_middle

        while (counting < N_aux)
            number_upper = state_upper & ((1 << spacing) - 1)        # This gets the last two numbers
            state_upper = state_upper >> spacing                     # And deletes it from the previous number
            bits_number_upper = countBits(number_upper)              # Bits in the two bits
            bits_state_upper = countBits(state_upper)                # Bits in the rest of the upper number

            if ((mod(bits_number_upper,2) == 1) && (mod(bits_state_upper,2)==1) )
                phase += 1
            end

            number_lower = state_lower & ((1 << spacing) - 1)        # This gets the last two numbers
            state_lower = state_lower >> spacing                     # And deletes it from the previous number
            bits_number_lower = countBits(number_lower)              # Bits in the two bits
            bits_state_lower = countBits(state_lower)                # Bits in the rest of the lower number 

            if ((mod(bits_number_lower,2) == 1) && (mod(bits_state_lower,2)==1) )
                phase += 1
            end

            counting += bits_number_upper + bits_number_lower

            outstate += number_lower*2^(half_L-1-2*i)*2^(half_L+1)+number_upper*2^(half_L-1-2*i)

            i+=1 
        end
        outstate += state_middle*2^(half_L-1)           # The m = 0 orbital
        return (outstate,(-1)^phase)
    end         
end


function basisRotY(stateList, half_L::Int, N::Int, rot::Int)
    basisMap = Dict()
    stateID = 0 
    basisRot = Int[]

    for state in stateList
        (state_new, factor) = reflectionBits(state,half_L,N)
        if state < state_new
            stateID += 1
            basisMap[state] = stateID
            push!(basisRot, state)
        end
        if state == state_new
            if factor == rot
                stateID += 1
                basisMap[state] = stateID
                push!(basisRot, state)
            end
        end
    end
    return (basisRot, basisMap)
end


function basisZ2rotY(stateList, half_L::Int, N::Int, parityZ2::Int,rot::Int)
    basisMap = Dict()
    stateID = 0 
    basisZ2Rot = Int[]

    for state in stateList

        (stateZ2, factorZ2) = swapBits(state)
        (stateRot, factorRot) = reflectionBits(state,half_L,N)
        (stateRotZ2, factorRotZ2) = reflectionBits(stateZ2,half_L,N)

        if ((state == stateZ2) && (state == stateRot))   # Here there's only one state
            if ((factorZ2 == parityZ2) && (factorRot == rot))
                stateID += 1
                basisMap[state] = stateID
                push!(basisZ2Rot, state)
            end 
        elseif (state == stateZ2)   # Here there are two states: one eigenstate of Z2 and the other (1+rot*Pi)
            if ((factorZ2 == parityZ2) && (state < stateRot))
                stateID += 1
                basisMap[state] = stateID
                push!(basisZ2Rot, state)
            end
        elseif (state == stateRot)   # Here there are two states: one eigenstate of Pi and the other (1+parityZ2*Z2)
            if ((factorRot == rot) && (state < stateZ2))
                stateID += 1
                basisMap[state] = stateID
                push!(basisZ2Rot, state)
            end
        elseif ((stateRot == stateZ2) && (state < stateZ2) && (parityZ2*factorZ2*rot*factorRot == 1)) # Here two states because Pi and Z2 maps to the same state
            stateID += 1                                                                              # Though careful because this could cancel each other.
            basisMap[state] = stateID
            push!(basisZ2Rot, state)
        
        elseif (state < min(stateZ2, stateRot, stateRotZ2))       # Here there are 4 states. 
            stateID += 1
            basisMap[state] = stateID
            push!(basisZ2Rot, state)

        elseif ((stateZ2 != stateRot) && (state == stateRotZ2))
            println("Here, we are having problems lol")
        end
    end
    return (basisZ2Rot, basisMap)
end


function stateRotZ2(state, half_L::Int, N::Int, parityZ2::Int,rot::Int)

    (stateZ2, factorZ2) = swapBits(state)
    (stateRot, factorRot) = reflectionBits(state,half_L,N)
    (stateRotZ2, factorRotZ2) = reflectionBits(stateZ2,half_L,N)
    nb = 1

    if ((state == stateZ2) && (state == stateRot))   # Here there's only one state and it is good
        nb = 1 
        if ((factorZ2 == parityZ2) && (factorRot == rot))
            fact = 1
            return (state, fact, nb)
        else 
            return (state, 0, nb)
        end 
        
    elseif (state == stateZ2)   # Here there are two states: one eigenstate of Z2 and the other (1+rot*Pi)
        nb = 2
        if (factorZ2 == parityZ2)
            if (state < stateRot)  # Here i am already at the representative
                fact = 1
                return (state, fact, nb)
            elseif (state > stateRot)
                # Here there's some factor of rot and return stateRot
                fact = rot*factorRot
                return (stateRot, fact, nb)
            end
        else 
            return (state, 0, nb)
        end 
            
    elseif (state == stateRot)   # Here there are two states: one eigenstate of Pi and the other (1+parityZ2*Z2)
        nb = 2 
        if (factorRot == rot)
            if (state < stateZ2)
                fact = 1 
                return (state, fact, nb)
            elseif (state > stateZ2)
                fact = parityZ2*factorZ2
                return (stateZ2, fact, nb)
            end
        else
            return (state, 0, nb)
        end
        
    elseif ((stateRot == stateZ2) && (parityZ2*factorZ2*rot*factorRot == 1))
        nb = 2 
        if (state < stateZ2)
            fact = 1 
            return (state, fact, nb)
        elseif (state > stateZ2)
            # this I am not sure
            fact = parityZ2*factorZ2
            return (stateZ2, fact, nb)
        end

    elseif (size(unique([state,stateZ2,stateRot,stateRotZ2]),1) == 4)
        nb = 4
        if (state < min(stateZ2,stateRot,stateRotZ2))
            fact = 1
            return (state, fact, nb)
        elseif (stateZ2 < min(state,stateRot,stateRotZ2))
            fact = parityZ2*factorZ2
            return (stateZ2, fact, nb)
        elseif (stateRot < min(state,stateZ2,stateRotZ2))
            fact = rot*factorRot
            return (stateRot, fact, nb)
        else
            fact = rot*parityZ2*factorZ2*factorRot
            return (stateRotZ2, fact, nb)
        end
        
    elseif ((stateZ2 != stateRot) && (state == stateRotZ2))
        println("Here, we are having problems lol")   # This may be unnecesary 
        
    else
        return (state, 0, nb)
    end
end

### Define magnetization in the other basis

function computeMagnetization(inputstate, stateList, basisMap, half_L::Int)
    mm = 0
    m2 = 0
    m4 = 0
    for state in stateList
        value_basis = abs(inputstate[basisMap[state]])^2
        if value_basis > 0 
            bits_state = countSignedBits(state)
            mm += 0.5*(bits_state)*value_basis
            m2 += value_basis*(0.5*bits_state)^2
            m4 += value_basis*(0.5*bits_state)^4
        end 
    end
    return (mm,m2,m4)
end


##################################### L2 functions ###########################################


function expvalTotAngularMomentum(inputstate, stateList, basisMap, N::Int, half_L::Int, mz::Int)
    
    L2 = 0 
    s = half_L - 1 

    for state in stateList
        value_basis = inputstate[basisMap[state]]

        L2 += (mz^2/4+mz/2)*value_basis^2

        for m1 in 1:s
            for m2 in 0:(s-1)
                factor = sqrt((2*s-2*m2)*(2*m2+2)*2*m1*(2*s-2*m1+2))/4

                if ((m1 != m2) && ((m2+1)!=(m1-1)))
                    ### up-up states
                    mask1 = 2^(2*m1+1)+2^(2*m1-1)
                    if ((state & mask1) == 2^(2*m1-1))
                        state_new = state ⊻ mask1
                        mask2 = 2^(2*m2+1)+2^(2*m2+3)
                        if ((state_new & mask2) == 2^(2*m2+3))
                            state_new2 = state_new ⊻ mask2
                            value_basis2 = inputstate[basisMap[state_new2]]
                            phase1 = state >> (2*m1+2)
                            totBits = countBits(phase1)
                            phase2 = state >> (2*m1)
                            totBits += countBits(phase2)-1
                            phase3 = state_new >> (2*m2+2)
                            totBits += countBits(phase3)
                            phase4 = state_new >> (2*m2+4)
                            totBits += countBits(phase4)

                            L2 += value_basis*value_basis2*factor*(-1)^totBits
                        end
                    end

                    ### down-down states
                    mask1 = 2^(2*m1)+2^(2*m1-2)
                    if ((state & mask1) == 2^(2*m1-2))
                        state_new = state ⊻ mask1
                        mask2 = 2^(2*m2)+2^(2*m2+2)
                        if ((state_new & mask2) == 2^(2*m2+2))
                            state_new2 = state_new ⊻ mask2
                            value_basis2 = inputstate[basisMap[state_new2]]
                            phase1 = state >> (2*m1+1)
                            totBits = countBits(phase1)
                            phase2 = state >> (2*m1-1)
                            totBits += countBits(phase2)-1
                            phase3 = state_new >> (2*m2+1)
                            totBits += countBits(phase3)
                            phase4 = state_new >> (2*m2+3)
                            totBits += countBits(phase4)

                            L2 += value_basis*value_basis2*factor*(-1)^totBits
                        end
                    end
                end

                ## up-down states
                mask1 = 2^(2*m1)+2^(2*m1-2)
                if ((state & mask1) == 2^(2*m1-2))
                    state_new = state ⊻ mask1
                    mask2 = 2^(2*m2+1)+2^(2*m2+3)
                    if ((state_new & mask2) == 2^(2*m2+3))
                        state_new2 = state_new ⊻ mask2
                        value_basis2 = inputstate[basisMap[state_new2]]
                        phase1 = state >> (2*m1+1)
                        totBits = countBits(phase1)
                        phase2 = state >> (2*m1-1)
                        totBits += countBits(phase2)-1
                        phase3 = state_new >> (2*m2+2)
                        totBits += countBits(phase3)
                        phase4 = state_new >> (2*m2+4)
                        totBits += countBits(phase4)

                        L2 += value_basis*value_basis2*factor*(-1)^totBits
                    end
                end 

                ### down-up states
                mask1 = 2^(2*m1+1)+2^(2*m1-1)
                if ((state & mask1) == 2^(2*m1-1))
                    state_new = state ⊻ mask1
                    mask2 = 2^(2*m2)+2^(2*m2+2)
                    if ((state_new & mask2) == 2^(2*m2+2))
                        state_new2 = state_new ⊻ mask2
                        value_basis2 = inputstate[basisMap[state_new2]]
                        phase1 = state >> (2*m1+2)
                        totBits = countBits(phase1)
                        phase2 = state >> (2*m1)
                        totBits += countBits(phase2)-1
                        phase3 = state_new >> (2*m2+1)
                        totBits += countBits(phase3)
                        phase4 = state_new >> (2*m2+3)
                        totBits += countBits(phase4)

                        L2 += value_basis*value_basis2*factor*(-1)^totBits
                    end
                end 
            end
        end
    end
    return L2
end


function L2matrix(stateList,basisMap,N::Int,half_L::Int, mz::Int)

    #@assert (mz == 0) "The value of mz is different from 0."

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    s = (half_L-1)
    spin = s/2

    for state in stateList
        idxA = basisMap[state]

        push!(cols,idxA)
        push!(rows,idxA)
        push!(entries,(mz^2/4+mz/2))

        # But this should give the same spectrum if mz is equal right? This is the identity on this subspace. I guess this is a good check


        for m1 in 1:s
            for m2 in 0:(s-1)
                factor = sqrt((2*s-2*m2)*(2*m2+2)*2*m1*(2*s-2*m1+2))/4

                if ((m1 != m2) && ((m2+1)!=(m1-1)))
                    ### up-up states
                    mask1 = 2^(2*m1+1)+2^(2*m1-1)
                    if ((state & mask1) == 2^(2*m1-1))
                        state_new = state ⊻ mask1
                        mask2 = 2^(2*m2+1)+2^(2*m2+3)
                        if ((state_new & mask2) == 2^(2*m2+3))
                            state_new2 = state_new ⊻ mask2
                            idxB = basisMap[state_new2]
                            
                            phase1 = state >> (2*m1+2)
                            totBits = countBits(phase1)
                            phase2 = state >> (2*m1)
                            totBits += countBits(phase2)-1
                            phase3 = state_new >> (2*m2+2)
                            totBits += countBits(phase3)
                            phase4 = state_new >> (2*m2+4)
                            totBits += countBits(phase4)

                            push!(cols,idxA)
                            push!(rows,idxB)
                            push!(entries,factor*(-1)^totBits)

                        end
                    end

                    ### down-down states
                    mask1 = 2^(2*m1)+2^(2*m1-2)
                    if ((state & mask1) == 2^(2*m1-2))
                        state_new = state ⊻ mask1
                        mask2 = 2^(2*m2)+2^(2*m2+2)
                        if ((state_new & mask2) == 2^(2*m2+2))
                            state_new2 = state_new ⊻ mask2
                            idxB = basisMap[state_new2]

                            phase1 = state >> (2*m1+1)
                            totBits = countBits(phase1)
                            phase2 = state >> (2*m1-1)
                            totBits += countBits(phase2)-1
                            phase3 = state_new >> (2*m2+1)
                            totBits += countBits(phase3)
                            phase4 = state_new >> (2*m2+3)
                            totBits += countBits(phase4)

                            push!(cols,idxA)
                            push!(rows,idxB)
                            push!(entries,factor*(-1)^totBits)
                        end
                    end
                end

                ## up-down states
                mask1 = 2^(2*m1)+2^(2*m1-2)
                if ((state & mask1) == 2^(2*m1-2))
                    state_new = state ⊻ mask1
                    mask2 = 2^(2*m2+1)+2^(2*m2+3)
                    if ((state_new & mask2) == 2^(2*m2+3))
                        state_new2 = state_new ⊻ mask2
                        idxB = basisMap[state_new2]

                        phase1 = state >> (2*m1+1)
                        totBits = countBits(phase1)
                        phase2 = state >> (2*m1-1)
                        totBits += countBits(phase2)-1
                        phase3 = state_new >> (2*m2+2)
                        totBits += countBits(phase3)
                        phase4 = state_new >> (2*m2+4)
                        totBits += countBits(phase4)

                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,factor*(-1)^totBits)
                    end
                end 

                ### down-up states
                mask1 = 2^(2*m1+1)+2^(2*m1-1)
                if ((state & mask1) == 2^(2*m1-1))
                    state_new = state ⊻ mask1
                    mask2 = 2^(2*m2)+2^(2*m2+2)
                    if ((state_new & mask2) == 2^(2*m2+2))
                        state_new2 = state_new ⊻ mask2
                        idxB = basisMap[state_new2]
                        
                        phase1 = state >> (2*m1+2)
                        totBits = countBits(phase1)
                        phase2 = state >> (2*m1)
                        totBits += countBits(phase2)-1
                        phase3 = state_new >> (2*m2+1)
                        totBits += countBits(phase3)
                        phase4 = state_new >> (2*m2+3)
                        totBits += countBits(phase4)

                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,factor*(-1)^totBits)
                    end
                end 
            end
        end
    end
    return sparse(cols,rows, entries)
end


########################################## Making the Hamiltonian ##########################################

function makeH(data::HIsing)
    N = data.N
    half_L = data.half_L
    h = data.h
    v00 = data.v00
    v11 = data.v11
    J00 = v00 #data.J00
    J11 = v11 #data.J11
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    s = (half_L-1)      # At half-filling, we know that N = 2spin+1. We compute s = 2spin = N-1. 
    spin = s/2               # And here, we compute the value of the spin.

    V0 = (4*spin+1) * v00   # Yin's paper, potential is V = \sum_l V_l (4spin-2l+1) wigner3j*wigner3j
    V1 = (4*spin-1) * v11   # Def: V0=(4spin+1)V_0, V1=(4spin-1)V_1 [including spin part on definition] 
    J0 = (4*spin+1) * J00
    J1 = (4*spin-1) * J11
                             # Setting V_1 = 1 and V_0 = 4.75 [value that Yin-Chen uses] 
    for state in stateList
        idxA = basisMap[state]

        if (h != 0)
            for i in 0:(half_L-1)
                mask_down = 2^(2*i)
                mask_up = 2^(2*i+1)

                mask = mask_down + mask_up 

                
                #### c^\dagger_down c_up ###
                if ((state & mask) == mask_up)
                    stateB = state ⊻ mask
                    idxB = basisMap[stateB]
                    push!(cols,idxA)
                    push!(rows,idxB)
                    push!(entries,-h)
                end

                #### c^\dagger_up c_down ####
                if ((state & mask) == mask_down)
                    stateB = state ⊻ mask
                    idxB = basisMap[stateB]
                    push!(cols,idxA)
                    push!(rows,idxB)
                    push!(entries,-h)
                end
            end
        end

        for m1 in 0:s
            for m2 in 0:s
                for m3 in 0:s
                    m4 = m1 + m2 - m3
                    if (0<= m4 <= s)
                        Vl = 0
                        Jl = 0
                        wignerl0 = wigner3j(Float64,spin,spin,s,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m4-spin,m3-spin,-m3-m4+s)
                        #wignerl0 += wigner3j(Float64,spin,spin,s,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m3-spin,m4-spin,-m3-m4+s)
                        wignerl1 = 0 
                        if (abs(-m1-m2+s) <= s-1) && (abs(-m3-m4+s) <= s-1)
                            wignerl1 = wigner3j(Float64,spin,spin,s-1,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m4-spin,m3-spin,-m3-m4+s)
                            #wignerl1 += wigner3j(Float64,spin,spin,s-1,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m3-spin,m4-spin,-m3-m4+s)
                        end
                        #println(wignerl0," ",wignerl1)
                        Vl += V0*wignerl0
                        Vl += V1*wignerl1
                        Jl += J0*wignerl0
                        Jl += J1*wignerl1
                        #println(Vl)
                        Vuu = (Vl - Jl)
                        Vud = Vl + Jl

                        if (Vl == Jl)
                            Vuu = 0
                        end

                        # There's the over 1/2 from going to LLL projection (Right)

                        if (Vud != 0)
                            annihilation_mask = 2^(2*m4+1) + 2^(2*m3)           
                            if ((state & annihilation_mask) == annihilation_mask)  
                                state_new = state ⊻ annihilation_mask              
                                creation_mask = 2^(2*m1+1) + 2^(2*m2)          
                                if ((state_new & creation_mask) == 0)               
                                    state_new2 = state_new ⊻ creation_mask           
                                    idxBB = basisMap[state_new2]                    
                                    ############ May be wrong this #################

                                    state_m3 = state >> (2*m3+1)
                                    state_new0 = state ⊻ 2^(2*m3)
                                    state_m4 = state_new0 >> (2*m4+2)
                                    state_m2 = state_new >> (2*m2+1)
                                    state_new1 = state_new ⊻ 2^(2*m2)
                                    state_m1 = state_new1 >> (2*m1+2)                  

                                    bitstot = countBits(state_m3)                 
                                    bitstot += countBits(state_m4)
                                    bitstot += countBits(state_m2)
                                    bitstot += countBits(state_m1)
                                    bitstot += 1

                                    factor = (-1)^bitstot
                                    
                                    push!(cols, idxA)
                                    push!(rows, idxBB)
                                    push!(entries, factor*Vud)  
                                end
                            end
                        end
                    end
                end
            end
        end                                              
        
    end
    return sparse(cols,rows, entries)
end


function makeH(data::HIsingZ2)
    N = data.N
    half_L = data.half_L
    parityZ2 = data.parityZ2
    h = data.h
    v00 = data.v00
    v11 = data.v11
    J00 = v00 #data.J00
    J11 = v11 #data.J11
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    s = (half_L-1)      # At half-filling, we know that N = 2spin+1. We compute s = 2spin = N-1. 
    spin = s/2               # And here, we compute the value of the spin.

    V0 = (4*spin+1) * v00   # Yin's paper, potential is V = \sum_l V_l (4spin-2l+1) wigner3j*wigner3j
    V1 = (4*spin-1) * v11   # Def: V0=(4spin+1)V_0, V1=(4spin-1)V_1 [including spin part on definition] 
    J0 = (4*spin+1) * J00
    J1 = (4*spin-1) * J11
                             # Setting V_1 = 1 and V_0 = 4.75 [value that Yin-Chen uses] 
    for state in stateList
        idxA = basisMap[state]

        ### This is to see how many states na there's in the representative
        na = 2
        (state1,_) = swapBits(state)
        if state == state1
            na = 1
        end

        if (h != 0)
            for i in 0:(half_L-1)
                mask_down = 2^(2*i)
                mask_up = 2^(2*i+1)
                mask = mask_down + mask_up 
                
                #### c^\dagger_down c_up ###
                if ((state & mask) == mask_up)
                    stateB = state ⊻ mask
                    (stateB, fact, nb) = stateZ2(stateB, parityZ2)
                    if (fact != 0)
                        idxB = basisMap[stateB]
                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,-h*fact*sqrt(na/nb))
                    end
                end

                #### c^\dagger_up c_down ####
                if ((state & mask) == mask_down)
                    stateB = state ⊻ mask
                    (stateB, fact, nb) = stateZ2(stateB, parityZ2)
                    if (fact != 0)
                        idxB = basisMap[stateB]
                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,-h*fact*sqrt(na/nb))
                    end
                end
            end
        end

        for m1 in 0:s
            for m2 in 0:s
                for m3 in 0:s
                    m4 = m1 + m2 - m3
                    if (0<= m4 <= s)
                        Vl = 0
                        Jl = 0
                        wignerl0 = wigner3j(Float64,spin,spin,s,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m4-spin,m3-spin,-m3-m4+s)
                        #wignerl0 += wigner3j(Float64,spin,spin,s,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m3-spin,m4-spin,-m3-m4+s)
                        wignerl1 = 0 
                        if (abs(-m1-m2+s) <= s-1) && (abs(-m3-m4+s) <= s-1)
                            wignerl1 = wigner3j(Float64,spin,spin,s-1,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m4-spin,m3-spin,-m3-m4+s)
                            #wignerl1 += wigner3j(Float64,spin,spin,s-1,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m3-spin,m4-spin,-m3-m4+s)
                        end
                        #println(wignerl0," ",wignerl1)
                        Vl += V0*wignerl0
                        Vl += V1*wignerl1
                        Jl += J0*wignerl0
                        Jl += J1*wignerl1
                        #println(Vl)
                        Vuu = (Vl - Jl)
                        Vud = Vl + Jl

                        if (Vl == Jl)
                            Vuu = 0
                        end

                        # There's the over 1/2 from going to LLL projection (Right)

                        if (Vud != 0)
                            annihilation_mask = 2^(2*m4+1) + 2^(2*m3)           
                            if ((state & annihilation_mask) == annihilation_mask)  
                                state_new = state ⊻ annihilation_mask              
                                creation_mask = 2^(2*m1+1) + 2^(2*m2)          
                                if ((state_new & creation_mask) == 0)               
                                    state_new2 = state_new ⊻ creation_mask

                                    (state_new2, fact, nb) = stateZ2(state_new2, parityZ2)

                                    if (fact != 0)
                                        
                                        idxBB = basisMap[state_new2]                    
                                        ############ May be wrong this #################

                                        state_m3 = state >> (2*m3+1)
                                        state_new0 = state ⊻ 2^(2*m3)
                                        state_m4 = state_new0 >> (2*m4+2)
                                        state_m2 = state_new >> (2*m2+1)
                                        state_new1 = state_new ⊻ 2^(2*m2)
                                        state_m1 = state_new1 >> (2*m1+2)                  

                                        bitstot = countBits(state_m3)                 
                                        bitstot += countBits(state_m4)
                                        bitstot += countBits(state_m2)
                                        bitstot += countBits(state_m1)
                                        bitstot += 1

                                        factor = (-1)^bitstot
                                    
                                        push!(cols, idxA)
                                        push!(rows, idxBB)
                                        push!(entries, fact*factor*Vud*sqrt(na/nb))
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end                                              
        
    end
    return sparse(cols,rows, entries)
end


function makeH(data::HIsingZ2Rot)
    N = data.N
    half_L = data.half_L
    parityZ2 = data.parityZ2
    rot = data.rot
    h = data.h
    v00 = data.v00
    v11 = data.v11
    J00 = v00 #data.J00
    J11 = v11 #data.J11
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    s = (half_L-1)      # At half-filling, we know that N = 2spin+1. We compute s = 2spin = N-1. 
    spin = s/2               # And here, we compute the value of the spin.

    V0 = (4*spin+1) * v00   # Yin's paper, potential is V = \sum_l V_l (4spin-2l+1) wigner3j*wigner3j
    V1 = (4*spin-1) * v11   # Def: V0=(4spin+1)V_0, V1=(4spin-1)V_1 [including spin part on definition] 
    J0 = (4*spin+1) * J00
    J1 = (4*spin-1) * J11
                             # Setting V_1 = 1 and V_0 = 4.75 [value that Yin-Chen uses] 
    for state in stateList
        idxA = basisMap[state]

        ### This is to see how many states na there's in the representative
        (state1,_) = swapBits(state)
        (state2,_) = reflectionBits(state, half_L, N)
        (state3,_) = reflectionBits(state1,half_L,N)
        na = size(unique([state,state1,state2,state3]),1)

        if (h != 0)
            for i in 0:(half_L-1)
                mask_down = 2^(2*i)
                mask_up = 2^(2*i+1)
                mask = mask_down + mask_up 
                
                #### c^\dagger_down c_up ###
                if ((state & mask) == mask_up)
                    stateB = state ⊻ mask
                    (stateB, fact, nb) = stateRotZ2(stateB, half_L, N, parityZ2,rot)  
                    if (fact != 0)
                        idxB = basisMap[stateB]
                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,-h*fact*sqrt(na/nb))
                    end
                end

                #### c^\dagger_up c_down ####
                if ((state & mask) == mask_down)
                    stateB = state ⊻ mask
                    (stateB, fact, nb) = stateRotZ2(stateB, half_L, N, parityZ2, rot)
                    if (fact != 0)
                        idxB = basisMap[stateB]
                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,-h*fact*sqrt(na/nb))
                    end
                end
            end
        end

        for m1 in 0:s
            for m2 in 0:s
                for m3 in 0:s
                    m4 = m1 + m2 - m3
                    if (0<= m4 <= s)
                        Vl = 0
                        Jl = 0
                        wignerl0 = wigner3j(Float64,spin,spin,s,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m4-spin,m3-spin,-m3-m4+s)
                        #wignerl0 += wigner3j(Float64,spin,spin,s,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m3-spin,m4-spin,-m3-m4+s)
                        wignerl1 = 0 
                        if (abs(-m1-m2+s) <= s-1) && (abs(-m3-m4+s) <= s-1)
                            wignerl1 = wigner3j(Float64,spin,spin,s-1,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m4-spin,m3-spin,-m3-m4+s)
                            #wignerl1 += wigner3j(Float64,spin,spin,s-1,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m3-spin,m4-spin,-m3-m4+s)
                        end
                        #println(wignerl0," ",wignerl1)
                        Vl += V0*wignerl0
                        Vl += V1*wignerl1
                        Jl += J0*wignerl0
                        Jl += J1*wignerl1
                        #println(Vl)
                        Vuu = (Vl - Jl)
                        Vud = Vl + Jl

                        if (Vl == Jl)
                            Vuu = 0
                        end

                        # There's the over 1/2 from going to LLL projection (Right)

                        if (Vud != 0)
                            annihilation_mask = 2^(2*m4+1) + 2^(2*m3)           
                            if ((state & annihilation_mask) == annihilation_mask)  
                                state_new = state ⊻ annihilation_mask              
                                creation_mask = 2^(2*m1+1) + 2^(2*m2)          
                                if ((state_new & creation_mask) == 0)               
                                    state_new2 = state_new ⊻ creation_mask

                                    (state_new2, fact, nb) = stateRotZ2(state_new2, half_L, N, parityZ2, rot)

                                    if (fact != 0)
                                        
                                        idxBB = basisMap[state_new2]                    
                                        ############ May be wrong this #################

                                        state_m3 = state >> (2*m3+1)
                                        state_new0 = state ⊻ 2^(2*m3)
                                        state_m4 = state_new0 >> (2*m4+2)
                                        state_m2 = state_new >> (2*m2+1)
                                        state_new1 = state_new ⊻ 2^(2*m2)
                                        state_m1 = state_new1 >> (2*m1+2)                  

                                        bitstot = countBits(state_m3)                 
                                        bitstot += countBits(state_m4)
                                        bitstot += countBits(state_m2)
                                        bitstot += countBits(state_m1)
                                        bitstot += 1

                                        factor = (-1)^bitstot
                                    
                                        push!(cols, idxA)
                                        push!(rows, idxBB)
                                        push!(entries, fact*factor*Vud*sqrt(na/nb))
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end                                              
        
    end
    return sparse(cols,rows, entries)
end

################################################################# Alternative Ising model ##################################################
function makeH(data::HIsingAlt)
    N = data.N
    half_L = data.half_L
    h = data.h
    g00 = data.g00
    g11 = data.g11
    #J00 = data.J00
    #J11 = data.J11
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    s = (half_L-1)      # At half-filling, we know that N = 2spin+1. We compute s = 2spin = N-1. 
    spin = s/2               # And here, we compute the value of the spin.
                             
    for state in stateList
        idxA = basisMap[state]

        if (h != 0)
            for i in 0:(half_L-1)
                mask_down = 2^(2*i)
                mask_up = 2^(2*i+1)

                mask = mask_down + mask_up 

                
                #### c^\dagger_down c_up ###
                if ((state & mask) == mask_up)
                    stateB = state ⊻ mask
                    idxB = basisMap[stateB]
                    push!(cols,idxA)
                    push!(rows,idxB)
                    push!(entries,-h)
                end

                #### c^\dagger_up c_down ####
                if ((state & mask) == mask_down)
                    stateB = state ⊻ mask
                    idxB = basisMap[stateB]
                    push!(cols,idxA)
                    push!(rows,idxB)
                    push!(entries,-h)
                end
            end
        end
        for lm in 0:s
            for m1 in 0:s
                for m2 in 0:s
                    for m3 in 0:s
                        m4 = m1 + m2 - m3
                        if ((0<= m4 <= s) && (abs(m1-m4)<= lm) && (abs(m2-m3)<= lm))
                            Ul = (g00 - g11*lm*(lm+1)/N)*N*(2*lm+1)/(4*pi) # It is better to put half_L instead of N.
                            Ul *= (-1)^(m1+m3)*wigner3j(Float64,spin,lm,spin,-spin,0,spin)^2 # There's no m1+m3+2spin because I am shifting m_i -> m_i-spin
                            Ul *= wigner3j(Float64,spin,lm,spin,m1-spin,m4-m1,-m4+spin)*wigner3j(Float64,spin,lm,spin,m2-spin,m3-m2,-m3+spin)

                            if (Ul != 0)
                                annihilation_mask = 2^(2*m4+1) + 2^(2*m3)           
                                if ((state & annihilation_mask) == annihilation_mask)  
                                    state_new = state ⊻ annihilation_mask              
                                    creation_mask = 2^(2*m1+1) + 2^(2*m2)          
                                    if ((state_new & creation_mask) == 0)               
                                        state_new2 = state_new ⊻ creation_mask           
                                        idxBB = basisMap[state_new2]                    
                                        ############ May be wrong this #################

                                        state_m3 = state >> (2*m3+1)
                                        state_new0 = state ⊻ 2^(2*m3)
                                        state_m4 = state_new0 >> (2*m4+2)
                                        state_m2 = state_new >> (2*m2+1)
                                        state_new1 = state_new ⊻ 2^(2*m2)
                                        state_m1 = state_new1 >> (2*m1+2)                  

                                        bitstot = countBits(state_m3)                 
                                        bitstot += countBits(state_m4)
                                        bitstot += countBits(state_m2)
                                        bitstot += countBits(state_m1)
                                        bitstot += 1

                                        factor = (-1)^bitstot
                                        
                                        push!(cols, idxA)
                                        push!(rows, idxBB)
                                        push!(entries, 4*factor*Ul)  
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end                                              
        
    end
    return sparse(cols,rows, entries)
end

####################################### Single band regularization #######################################

function makeH(data::HNewIsing)
    N = data.N
    half_L = data.half_L
    radius = data.radius
    h = data.h
    g00 = data.g00
    g11 = data.g11
    #J00 = data.J00
    #J11 = data.J11
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    lm_max = half_L-1      # This is the max value of the lm corresponding to U_l which is from [0,lm_max]
    lorb = div(lm_max,2)   # The value of the single band lorb we are projecting over.
                
    for state in stateList
        idxA = basisMap[state]

        if (h != 0)
            for i in 0:(half_L-1)
                mask_down = 2^(2*i)
                mask_up = 2^(2*i+1)

                mask = mask_down + mask_up 
   
                #### c^\dagger_down c_up ###
                if ((state & mask) == mask_up)
                    stateB = state ⊻ mask
                    idxB = basisMap[stateB]
                    push!(cols,idxA)
                    push!(rows,idxB)
                    push!(entries,-h)
                end

                #### c^\dagger_up c_down ####
                if ((state & mask) == mask_down)
                    stateB = state ⊻ mask
                    idxB = basisMap[stateB]
                    push!(cols,idxA)
                    push!(rows,idxB)
                    push!(entries,-h)
                end
            end
        end

        for lm in 0:2:lm_max
            for m1 in 0:lm_max
                for m2 in 0:lm_max
                    for m3 in 0:lm_max
                        m4 = m1 + m2 - m3
                        if ((0<= m4 <= lm_max) && (abs(m1-m4)<= lm) && (abs(m2-m3)<= lm))
                            Ul = (g00 - g11*lm*(lm+1)/radius^2)/N 
                            Ul *= (-1)^(m1+m3)*(2*lm+1)*N^2/(4*pi)
                            Ul *= wigner3j(Float64,lm,lorb,lorb,m1-m4,-m1+lorb,m4-lorb)*wigner3j(Float64,lm,lorb,lorb,m2-m3,-m2+lorb,m3-lorb)
                            Ul *= wigner3j(Float64,lm,lorb,lorb,0,0,0)^2
                            
                            # There's the over 1/2 from going to LLL projection (Right)

                            if (Ul != 0)
                                annihilation_mask = 2^(2*m4+1) + 2^(2*m3)           
                                if ((state & annihilation_mask) == annihilation_mask)  
                                    state_new = state ⊻ annihilation_mask              
                                    creation_mask = 2^(2*m1+1) + 2^(2*m2)          
                                    if ((state_new & creation_mask) == 0)               
                                        state_new2 = state_new ⊻ creation_mask           
                                        idxBB = basisMap[state_new2]                    
                                        ############ May be wrong this #################

                                        state_m3 = state >> (2*m3+1)
                                        state_new0 = state ⊻ 2^(2*m3)
                                        state_m4 = state_new0 >> (2*m4+2)
                                        state_m2 = state_new >> (2*m2+1)
                                        state_new1 = state_new ⊻ 2^(2*m2)
                                        state_m1 = state_new1 >> (2*m1+2)                  

                                        bitstot = countBits(state_m3)                 
                                        bitstot += countBits(state_m4)
                                        bitstot += countBits(state_m2)
                                        bitstot += countBits(state_m1)
                                        bitstot += 1
    
                                        factor = (-1)^bitstot
                                        
                                        push!(cols, idxA)
                                        push!(rows, idxBB)
                                        push!(entries, 4*factor*Ul)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end                                              
        
    end
    return sparse(cols,rows, entries)
end

## This one I haven't change
function makeH(data::HNewIsingZ2)
    N = data.N
    half_L = data.half_L
    radius = data.radius
    parityZ2 = data.parityZ2
    h = data.h
    g00 = data.g00
    g11 = data.g11
    #J00 = data.J00
    #J11 = data.J11
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    lm_max = half_L-1
    lorb = div(lm_max,2)
    
    
    for state in stateList
        idxA = basisMap[state]

        ### This is to see how many states na there's in the representative
        na = 2
        (state1,_) = swapBits(state)
        if state == state1
            na = 1
        end

        if (h != 0)
            for i in 0:(half_L-1)
                mask_down = 2^(2*i)
                mask_up = 2^(2*i+1)
                mask = mask_down + mask_up 
                
                #### c^\dagger_down c_up ###
                if ((state & mask) == mask_up)
                    stateB = state ⊻ mask
                    (stateB, fact, nb) = stateZ2(stateB, parityZ2)
                    if (fact != 0)
                        idxB = basisMap[stateB]
                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,-h*fact*sqrt(na/nb))
                    end
                end

                #### c^\dagger_up c_down ####
                if ((state & mask) == mask_down)
                    stateB = state ⊻ mask
                    (stateB, fact, nb) = stateZ2(stateB, parityZ2)
                    if (fact != 0)
                        idxB = basisMap[stateB]
                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,-h*fact*sqrt(na/nb))
                    end
                end
            end
        end

        for lm in 0:2:lm_max
            for m1 in 0:lm_max
                for m2 in 0:lm_max
                    for m3 in 0:lm_max
                        m4 = m1 + m2 - m3
                        if ((0<= m4 <= lm_max) && (abs(m1-m4)<= lm) && (abs(m2-m3)<= lm))
                            Ul = (g00 - g11*lm*(lm+1)/(radius^2))/N
                            Ul *= (-1)^(m1+m3)*wigner3j(Float64,lm,lorb,lorb,0,0,0)^2 
                            Ul *= wigner3j(Float64,lm,lorb,lorb,m1-m4,-m1+lorb,m4-lorb)*wigner3j(Float64,lm,lorb,lorb,m2-m3,-m2+lorb,m3-lorb)
                            Ul *= (2*lm+1)*(2*lorb+1)^2/(4*pi)

                        # There's the over 1/2 from going to LLL projection (Right)

                            if (Ul != 0)
                                annihilation_mask = 2^(2*m4+1) + 2^(2*m3)           
                                if ((state & annihilation_mask) == annihilation_mask)  
                                    state_new = state ⊻ annihilation_mask              
                                    creation_mask = 2^(2*m1+1) + 2^(2*m2)          
                                    if ((state_new & creation_mask) == 0)               
                                        state_new2 = state_new ⊻ creation_mask
    
                                        (state_new2, fact, nb) = stateZ2(state_new2, parityZ2)
    
                                        if (fact != 0)
                                            
                                            idxBB = basisMap[state_new2]                    
                                            ############ May be wrong this #################
    
                                            state_m3 = state >> (2*m3+1)
                                            state_new0 = state ⊻ 2^(2*m3)
                                            state_m4 = state_new0 >> (2*m4+2)
                                            state_m2 = state_new >> (2*m2+1)
                                            state_new1 = state_new ⊻ 2^(2*m2)
                                            state_m1 = state_new1 >> (2*m1+2)                  

                                            bitstot = countBits(state_m3)                 
                                            bitstot += countBits(state_m4)
                                            bitstot += countBits(state_m2)
                                            bitstot += countBits(state_m1)
                                            bitstot += 1

                                            factor = (-1)^bitstot
                                    
                                            push!(cols, idxA)
                                            push!(rows, idxBB)
                                            push!(entries, 4*fact*factor*Ul*sqrt(na/nb))
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end                                                 
    end
    return sparse(cols,rows, entries)
end



function makeH(data::HNewIsingZ2Rot)
    N = data.N
    half_L = data.half_L
    radius = data.radius
    parityZ2 = data.parityZ2
    rot = data.rot
    h = data.h
    g00 = data.g00
    g11 = data.g11
    #J00 = data.J00
    #J11 = data.J11
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    lm_max = half_L-1
    lorb = div(lm_max,2)
    
    for state in stateList
        idxA = basisMap[state]

        ### This is to see how many states na there's in the representative
        (state1,_) = swapBits(state)
        (state2,_) = reflectionBits(state, half_L, N)
        (state3,_) = reflectionBits(state1,half_L,N)
        na = size(unique([state,state1,state2,state3]),1)

        if (h != 0)
            for i in 0:(half_L-1)
                mask_down = 2^(2*i)
                mask_up = 2^(2*i+1)
                mask = mask_down + mask_up 
                
                #### c^\dagger_down c_up ###
                if ((state & mask) == mask_up)
                    stateB = state ⊻ mask
                    (stateB, fact, nb) = stateRotZ2(stateB, half_L, N, parityZ2,rot)  
                    if (fact != 0)
                        idxB = basisMap[stateB]
                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,-h*fact*sqrt(na/nb))
                    end
                end

                #### c^\dagger_up c_down ####
                if ((state & mask) == mask_down)
                    stateB = state ⊻ mask
                    (stateB, fact, nb) = stateRotZ2(stateB, half_L, N, parityZ2, rot)
                    if (fact != 0)
                        idxB = basisMap[stateB]
                        push!(cols,idxA)
                        push!(rows,idxB)
                        push!(entries,-h*fact*sqrt(na/nb))
                    end
                end
            end
        end

        for lm in 0:2:lm_max
            for m1 in 0:lm_max
                for m2 in 0:lm_max
                    for m3 in 0:lm_max
                        m4 = m1 + m2 - m3
                        if ((0<= m4 <= lm_max) && (abs(m1-m4)<= lm) && (abs(m2-m3)<= lm))
                            Ul = (g00 - g11*lm*(lm+1)/radius^2)/N 
                            Ul *= (-1)^(m1+m3)*(2*lm+1)*N^2/(4*pi)
                            Ul *= wigner3j(Float64,lm,lorb,lorb,m1-m4,-m1+lorb,m4-lorb)*wigner3j(Float64,lm,lorb,lorb,m2-m3,-m2+lorb,m3-lorb)
                            Ul *= wigner3j(Float64,lm,lorb,lorb,0,0,0)^2

                        # There's the over 1/2 from going to LLL projection (Right)

                            if (Ul != 0)
                                annihilation_mask = 2^(2*m4+1) + 2^(2*m3)           
                                if ((state & annihilation_mask) == annihilation_mask)  
                                    state_new = state ⊻ annihilation_mask              
                                    creation_mask = 2^(2*m1+1) + 2^(2*m2)          
                                    if ((state_new & creation_mask) == 0)               
                                        state_new2 = state_new ⊻ creation_mask
    
                                        (state_new2, fact, nb) = stateRotZ2(state_new2, half_L, N, parityZ2, rot)

                                        if (fact != 0)
                                        
                                            idxBB = basisMap[state_new2]                    
                                            ############ May be wrong this #################

                                            state_m3 = state >> (2*m3+1)
                                            state_new0 = state ⊻ 2^(2*m3)
                                            state_m4 = state_new0 >> (2*m4+2)
                                            state_m2 = state_new >> (2*m2+1)
                                            state_new1 = state_new ⊻ 2^(2*m2)
                                            state_m1 = state_new1 >> (2*m1+2)                  

                                            bitstot = countBits(state_m3)                 
                                            bitstot += countBits(state_m4)
                                            bitstot += countBits(state_m2)
                                            bitstot += countBits(state_m1)
                                            bitstot += 1

                                            factor = (-1)^bitstot
                                    
                                            push!(cols, idxA)
                                            push!(rows, idxBB)
                                            push!(entries, 4*fact*factor*Ul*sqrt(na/nb))
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end                                              
        
    end
    return sparse(cols,rows, entries)
end