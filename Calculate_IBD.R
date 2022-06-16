library(AGHmatrix)
ped <- read.csv("/home/jordan/pop-model/temp/parents.csv", header = F)
Amat <- Amatrix(data = ped)
write.csv(Amat,"/home/jordan/pop-model/temp/sim_Amat.csv",row.names=F,quote = F)