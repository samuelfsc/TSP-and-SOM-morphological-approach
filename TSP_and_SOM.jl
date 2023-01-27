using Images, ImageMorphology
using Statistics, StatsBase, Distances
using LinearAlgebra
using ProgressMeter
using TravelingSalesmanHeuristics
using TravelingSalesmanExact, GLPK


# Lexicographical ordering with red, green and blue.
function h_rgb(img)
    return channelview(img)[1,:,:]+channelview(img)[2,:,:]/256+channelview(img)[3,:,:]/(256*256)
end

function hMM(img, h_function, mm_operator)
    n1,n2 = size(img)
    img_h = h_function(img)
    b = sortperm(img_h[:])
    im_latt = Array{Int64,1}(undef, n1*n2)
    im_latt[b] = 1:n1*n2
    im_latt = reshape(im_latt,n1,n2)
    img_mm = mm_operator(im_latt)
    return reshape(img[b[img_mm[:]]],n1,n2)
end

function mydilate(img)
    # Cross-structuring element!
    d1 = dilate(img,1)
    d2 = dilate(img,2)
    return max.(d1,d2)
end

function toy_2gray()
    I = zeros(RGB{N0f8}, 10, 5)
    gr = RGB(100/255,100/255,100/255)
    I[3,3] = gr
    I[7,3] = gr
    I[:,4:5] .= RGB(0,0,1)
    return I
end

#TSP
function hMMp(img, mm_operator)
    n1, n2 = size(img)
    m = n1*n2
    Im = float.(img)
    Cities = reshape(channelview(Im), 3, m)
    Cost_matrix = [norm(Cities[:,i]-Cities[:,j]) for i = 1:m, j = 1:m]
    vector_norm = Array{Float32,1}(undef, m)
    for i = 1:m
        vector_norm[i] = norm(Cities[:,i])
    end
    Tour, Cost = farthest_insertion(Cost_matrix, firstcity = argmin(vector_norm), do2opt = true)
    pop!(Tour)
    b = sortperm(Tour[:])
    img_latt = reshape(b, n1, n2)
    img_mm = mm_operator(img_latt)
    return reshape(img[Tour[img_mm[:]]], n1, n2)
end

function hMMpexact(img, mm_operator)
    n1, n2 = size(img)
    m = n1*n2
    Im = float.(img)
    Cities = reshape(channelview(Im), 3, m)
    Cost_matrix = [norm(Cities[:,i]-Cities[:,j]) for i = 1:m, j = 1:m]
    set_default_optimizer!(GLPK.Optimizer)
    Tour, Cost = get_optimal_tour(Cost_matrix)
    b = sortperm(Tour[:])
    img_latt = reshape(b, n1, n2)
    img_mm = mm_operator(img_latt)
    return reshape(img[Tour[img_mm[:]]], n1, n2)
end


function espectro(img)
    n1, n2 = size(img)
    m = n1*n2
    Im = float.(img)
    Cities = reshape(channelview(Im), 3, m)
    Cost_matrix = [norm(Cities[:,i]-Cities[:,j]) for i = 1:m, j = 1:m]
    vector_norm = Array{Float32,1}(undef, m)
    for i = 1:m
        vector_norm[i] = norm(Cities[:,i])
    end
    Tour, Cost = farthest_insertion(Cost_matrix, firstcity = argmin(vector_norm), do2opt = true)
    pop!(Tour)
    return img[Tour[:]]
end    

function espectro_exact(img)
    n1, n2 = size(img)
    m = n1*n2
    Im = float.(img)
    Cities = reshape(channelview(Im), 3, m)
    Cost_matrix = [norm(Cities[:,i]-Cities[:,j]) for i = 1:m, j = 1:m]
    set_default_optimizer!(GLPK.Optimizer)
    Tour, Cost = get_optimal_tour(Cost_matrix)
    return img[Tour[:]]
end

#SOM
function som(img, m = 1000, itMax = 1.e5)
    n1, n2 = size(img)
    D = reshape(channelview(img), 3, n1*n2)'
    L = collect(1:n1*n2)
    A = range(start = 0, stop = 1, length = m) 
    W = hcat(A,A,A)
    
    #Constantes para vizinhança hji
    sigma_zero = m/2
    sigma_final = 0.1
    sigma_ratio = sigma_final/sigma_zero
    
    #Constantes da Taxa de Aprendizado: eta(n)
    eta_zero = 0.5
    eta_final = 0.01
    eta_ratio = eta_final/eta_zero
    
    for it = 0:(itMax)
        j = rand(L)
        dist =  sqrt.((W[:,1].-D[j,1]).^2 + (W[:,2].-D[j,2]).^2 + (W[:,3].-D[j,3]).^2)
        v = argmin(dist)
        hji = zeros(m)
        dji = zeros(m)
        sigma = sigma_zero*(sigma_ratio^(it/itMax))
        k2 = -1/(2*sigma^2)
        for i = 1:v
            k = v - i + 1
            dji[k] = norm(k-v)
            hji[k] =  exp(k2*dji[k]^2)
        end
        for i = v+1:m
            dji[i] = norm(i-v)
            hji[i] = exp(k2*dji[i]^2)
        end
        # Parâmetro da Taxa de Aprendizado: eta(n)
        eta = eta_zero*(eta_ratio^(it/itMax))
        #println([it,eta])
        W = W + (eta*hji).*(D[j,:]'.-W)
    end
    return W
end

function h_som(img, W)
    n1, n2 = size(img)
    M = n1*n2
    vet = zeros(M)
    imagem = reshape(channelview(img), 3, M)'
    w = size(W)[1]
    for i = 1:M
        distancia = zeros(w)
        for j = 1:w
            distancia[j] = norm(imagem[i,:] - W[j,:])
        end
        vet[i] = argmin(distancia)
    end
    return reshape(vet, n1, n2)
end


function hMM_som(img, W, mm_operator)
    n1, n2 = size(img)
    img_h = h_som(img, W)
    b = sortperm(img_h[:])
    im_latt = Array{Int64,1}(undef, n1*n2)
    im_latt[b] = 1:n1*n2
    im_latt = reshape(im_latt,n1,n2)
    img_mm = mm_operator(im_latt)
    return reshape(img[b[img_mm[:]]],n1,n2)
end
