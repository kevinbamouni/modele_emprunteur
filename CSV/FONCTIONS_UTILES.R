# library(Hmisc)
# library(DT)
library(data.table)
# library(datapasta)
# library(here)
# library(varhandle)
# library(dplyr)
# library(rlist)
# library(MASS)
# data(painters)

###################################################################################
###################################################################################

df2excel1 = function (x) {
  tempFilePath = paste(tempfile(), ".csv")
  tempPath = dirname(tempFilePath)
  preferredFile = paste(deparse(substitute(x)), ".csv", sep = "")
  preferredFilePath = file.path(tempPath, preferredFile)
  
  if(length(dim(x))>2){
    stop('Too many dimensions')
  }
  if(is.null(dim(x))){
    x = as.data.frame(x)
  }
  if (is.null(rownames(x))) {
    tmp = 1:nrow(x)
  }else {
    tmp = rownames(x)
  }
  rownames(x) = NULL
  x = data.frame(RowLabels = tmp, x)
  WriteAttempt = try(
    write.table(x, file=preferredFilePath, quote=TRUE, sep=",", na="",
                row.names=FALSE, qmethod="double"),
    silent = TRUE)
  if ("try-error" %in% class(WriteAttempt)) {
    write.table(x, file=tempFilePath, , quote=TRUE, sep=",", na="",
                row.names=FALSE, qmethod="double")
    shell.exec(tempFilePath)
  } else {
    shell.exec(preferredFilePath)
  }
}


#df2excel1(painters)
##C:\Users\clark.djilo\AppData\Local\Temp\RtmpkRJrr8

###################################################################################
###################################################################################

df2excel = function (x) {
  tempFilePath = paste(tempfile(), ".csv")
  tempPath = dirname(tempFilePath)
  preferredFile = paste(deparse(substitute(x)), ".csv", sep = "")
  preferredFilePath = file.path(tempPath, preferredFile)
  
  if(length(dim(x))>2){
    stop('Too many dimensions')
  }
  if(is.null(dim(x))){
    x = as.data.frame(x)
  }
  if (is.null(rownames(x))) {
    tmp = 1:nrow(x)
  }else {
    tmp = rownames(x)
  }
  rownames(x) = NULL
  x = data.frame(RowLabels = tmp, x)
  WriteAttempt = try(
    data.table::fwrite(x, file=preferredFilePath, quote=TRUE),
    silent = TRUE)
  if ("try-error" %in% class(WriteAttempt)) {
    data.table::fwrite(x, file=tempFilePath, quote=TRUE)
    shell.exec(tempFilePath)
  } else {
    shell.exec(preferredFilePath)
  }
}

#df2excel2(painters)
##C:\Users\clark.djilo\AppData\Local\Temp\RtmpkRJrr8

###################################################################################
###################################################################################

# #DT
# #datatable(painters, filter = 'top')
# 
# #Hmisc
# Cs(Firefox, Chrome, Edge, Safari, InternetExplorer, Opera)           
# 
# #datapasta
# #vector_paste("f,g,h,i")
# 
# noquote(c("Firefox", "Chrome", "Edge", "Safari", "Internet Explorer", "Opera"))
# 
# 
# #####
# 
# data(iris)
# #str(iris)
# 
# #varhandle
# iris <-unfactor(iris)
# #str(iris)
# 
# #####
# #dplyr
# filter(painters, Colour >= 5 & School == 'A')
# 
# #####
# 
# x <- data.frame(a=1:3,type=c('A','C','B'))
# a = list.parse(x)
# #class(a)
# 
# 
# # y <- matrix(rnorm(1000),ncol=5)
# # rownames(y) <- paste0('item',1:nrow(x))
# # colnames(y) <- c('a','b','c','d','e')
# # b = list.parse(y)
# # class(b)
# 
# 
# z <- '
#  a:
#    type: x
#    class: A
#    registered: yes
#  '
# c = list.parse(z, type='yaml')
# #class(c)
# 
# 
# #rlist
# #####
# 
# x <- list(p1 = list(type='A',score=list(c1=10,c2=8)),
#           p2 = list(type='B',score=list(c1=9,c2=9)),
#           p3 = list(type='B',score=list(c1=9,c2=7)))
# 
# list.filter(x, type=='B')
# list.filter(x, min(score$c1, score$c2) >= 8)
# list.filter(x, type=='B', score$c2 >= 8)
# 
# 
# #rlist
# #####
# people <- list.parse('
#    Ken:
#      name: Ken
#      age: 24
#      interests: [reading, coding]
#      friends: [James, Ashley]
#    James:
#      name: James
#      age: 23
#      interests: [reading, movie, hiking]
#      friends: [Ken, David]
#    Ashley:
#      name: Ashley
#      age: 25
#      interests: [movies, music, reading]
#      friends: [Ken, David]
#    David:
#      name: David
#      age: 24
#      interests: [coding, hiking]
#      friends: [Ashley, James]
#  ',type = "yaml")
# #str(people)
# 
# 
# list.mapv(list.filter(people, age >= 25), name) 
# 
# # apply functions
# #################


# # recherchev
# #################

# # load sample data from Q
# hous <- read.table(header = TRUE, 
#                    stringsAsFactors = FALSE, 
#                    text="HouseType HouseTypeNo
# Semi            1
# Single          2
# Row             3
# Single          2
# Apartment       4
# Apartment       4
# Row             3")
# 
# # create a toy large table with a 'HouseType' column 
# # but no 'HouseTypeNo' column (yet)
# largetable <- data.frame(HouseType = as.character(sample(unique(hous$HouseType), 1000, replace = TRUE)), stringsAsFactors = FALSE)
# 
# # create a lookup table to get the numbers to fill
# # the large table
# lookup <- unique(hous)
# 
# head(lookup)
# head(largetable)
# 
# largetable$HouseTypeNo <- with(lookup,HouseTypeNo[match(largetable$HouseType,HouseType)])



#cat("Parent level \n \t Child level \n \t \t Double Child \n \t Child \n Parent level")


# Parent level 
#  Child level 
#   Double Child 
#  Child 
# Parent level


# define.catt <- function(ntab = NULL){
#   catt <- function(input = NULL){
#     cat(paste0(paste(rep("\t", ntab), collapse = ""), input))
#   }
#   return(catt)
# }
# 
# 
# catt <- define.catt(ntab = 1)
# catt("hi")
# hi
# catt <- define.catt(ntab = 2)
# catt("hi")
# hi





