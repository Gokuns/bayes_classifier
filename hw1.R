
# read data into memory
data_set <- read.csv("C:\\Users\\Goko\\Desktop\\hw01\\hw01_data_set_images.csv")
label_set <- read.csv("C:\\Users\\Goko\\Desktop\\hw01\\hw01_data_set_labels.csv")

# selecting the training data
train_a <- data_set[c(1:25),]
train_b <- data_set[c(40:64),]
train_c <- data_set[c(79:103),]
train_d <- data_set[c(118:142),]
train_e <- data_set[c(157:181),]

test_a <- data_set[c(26:39),]
test_b <- data_set[c(65:78),]
test_c <- data_set[c(104:117),]
test_d <- data_set[c(143:156),]
test_e <- data_set[c(182:195),]

label_a <- as.numeric(label_set[c(1:25),])
label_b <- as.numeric(label_set[c(40:64),])
label_c <- as.numeric(label_set[c(79:103),])
label_d <- as.numeric(label_set[c(118:142),])
label_e <- as.numeric(label_set[c(157:181),])


safelog <- function(x){
  if(x==0){
    return(0)
  }
  else{
    return(log(x))
  }
}


pcd <- function(x,y){
  a <- lenght(c)
  return (sapply(X = 1:25, FUN= function(c){mean(x[y==c])}))
}

pcd_a <- pcd(train_a, label_a)

abc <- sapply(X=1:320, FUN= function(c) {train_a[safelog(sapply(X = 1:25, FUN= function(c){mean(train_a[label_a==c])}))] + (1-train_a)[safelog(1-(sapply(X = 1:25, FUN= function(c){mean(x[label_a==c])})))]})
