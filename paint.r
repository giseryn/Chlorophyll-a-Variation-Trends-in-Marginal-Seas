install.packages("GGally")
library(GGally)

AE <-read.table("AE.csv", header=T,sep = ",",row.names = 1)
BS <-read.table("BS.csv", header=T,sep = ",",row.names = 1)
ECS <-read.table("ECS.csv", header=T,sep = ",",row.names = 1)
ECW <-read.table("ECW.csv", header=T,sep = ",",row.names = 1)
GM <-read.table("GM.csv", header=T,sep = ",",row.names = 1)



ggpairs(AE)
ggsave("amazon_corr.png",width = 5, height = 5, dpi = 300)
ggpairs(BS)
ggsave("bohai_corr.png",width = 5, height = 5, dpi = 300)
ggpairs(ECS)
ggsave("ECS_corr.png",width = 5, height = 5, dpi = 300)
ggpairs(ECW)
ggsave("ECW_corr.png",width = 5, height = 5, dpi = 300)
ggpairs(GM)
ggsave("GM_corr.png",width = 5, height = 5, dpi = 300)

