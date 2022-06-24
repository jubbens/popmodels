library(AGHmatrix)
ped <- read.csv("./temp/parents.csv", header = F)
Amat <- Amatrix(data = ped)
write.csv(Amat,"./temp/sim_Amat.csv",row.names=F,quote = F)
