# Functions - Mercurio
getV0 <- function(vol_0, rho_W, tStructure, i){
  d_sum = 0
  for(j in 1:i){
    for(k in 1:i){
      d_sum = d_sum + vol_0[j] * vol_0[k] * rho_W[j, k] * (min(tStructure[j], tStructure[k]) / tStructure[i])
    }
  }
  return(sqrt(d_sum))
}

getNu <- function(vol_0, rho_W, rho_V, volOfvol, tStructure, i){
  d_sum = 0
  V0 <- getV0(vol_0, rho_W, tStructure, i)
  for(j in 1:i){
    for(k in 1:i){
      d_sum = d_sum + (vol_0[j] / V0) * (vol_0[k] / V0) * rho_W[j, k] * rho_V[j, k] * volOfvol[j] *
        volOfvol[k] * (min(tStructure[j], tStructure[k]) / tStructure[i])^2
    }
  }
  return(sqrt(d_sum))
}

getRho <- function(vol_0, rho_W, rho_V, rho_VW, volOfVol, tStructure, i){
  t_sum = 0
  b_sum = 0
  V0 <- getV0(vol_0, rho_W, tStructure, i)
  for(j in 1:i){
    for(k in 1:i){
      for(l in 1:i){
        t_sum = t_sum + (vol_0[j] / V0) * (vol_0[k] / V0) * (vol_0[l] / V0) * rho_W[j, k] * rho_VW[j, l] * volOfVol[j]
        for(h in 1:i){
          b_sum = b_sum + (vol_0[j] / V0) * (vol_0[k] / V0) * (vol_0[l] / V0) * (vol_0[h] / V0) * 
            rho_W[j, k] * rho_W[l, h] * volOfVol[j] * volOfVol[l] * rho_V[j, l]
        }
      }
    }
  }
  return(t_sum / sqrt(b_sum))
}

# SABR Funciton
getSabrX <- function(z, rho){
  out <- log((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
  return(out)
}
getSabrPrice <- function(f, K, alpha, v, rho, mat){
  z <- (v / alpha) * log(f / K)
  #print(z)
  #print(getSabrX(z, rho))
  sig <- alpha * (1 + (alpha^2 + 0.25*rho*v*alpha + (2 - 3*rho^2 * v^2) / 24) * mat) * 
    (z / getSabrX(z, rho))
  d1 <- (log(f/K) + 0.5 * sig^2 * mat) / (sig * sqrt(mat))
  d2 <- (log(f/K) - 0.5 * sig^2 * mat) / (sig * sqrt(mat))
  #print(d1)
  #print(d2)
  price <- (f * pnorm(d1) - K * pnorm(d2))
  return(price)
}
