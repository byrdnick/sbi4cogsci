
rm(list=ls())

n=100   # total number of coin flips
S=24     # total number of subjects

true=list()
true$alpha=10
true$beta=15

data=list()

p=numeric(S)
for(j in 1:S){
  p[j]=rbeta(1,true$alpha,true$beta)   # probability of single-trial success
  data[[j]]=rbinom(n,size=1,prob=p[j])
}
true$p=p

x=sapply(data,function(x)sum(x))

################################################################# inference

# define a posterior distribution
log.dens.like=function(x,use.data){
  like=dbinom(use.data,size=1,prob=x,log=TRUE)
  return(sum(like))
}

log.dens.prior=function(x,alpha,beta){
  dbeta(x=x,shape1=alpha,shape2=beta,log=TRUE)
}
  
log.dens.post=function(x,alpha,beta,use.data){
  out=log.dens.like(x=x,use.data=use.data) + log.dens.prior(x=x,alpha=alpha,beta=beta)
  if(is.na(out))out=-Inf
  return(out)
}  

# we can evaluate the unnormalized posterior for different values of p
sub=1
log.dens.like(.4,use.data=data[[sub]])
log.dens.prior(.4,alpha=true$alpha,beta=true$beta)

xs=seq(0,1,.01)
lxs=length(xs)
ys=numeric(lxs)
for(i in 1:lxs){
ys[i]=log.dens.post(xs[i],alpha=true$alpha,beta=true$beta,use.data=data[[sub]])
}

# plot of posterior (on log scale)
plot(xs,ys)

################################################################# posterior approximation, method 1: assuming known alpha and beta

require(msm)

n.iter=1000    # total number of iterations
pset=lpset=matrix(NA,n.iter,S)
tune=.01

# initialize
# informed starting points
#pset[1,]=sapply(data,function(x)mean(x))

# 'your guess is as good as mine' starting points
pset[1,]=rbeta(S,1,1)

for(j in 1:S){
  lpset[1,j]=log.dens.post(pset[1,j],use.data=data[[j]],alpha=true$alpha,beta=true$beta)
}

# loop
for(t in 2:n.iter){
  pstar=rnorm(S,pset[t-1,],tune) # symmetric transition kernel
  lpstar=numeric(S)
  for(j in 1:S){
    lpstar[j]=log.dens.post(pstar[j],use.data=data[[j]],alpha=true$alpha,beta=true$beta)
    a=exp(lpstar[j]-lpset[t-1,j]) # note that proposal density is in the numerator
    if(runif(1)<a){ 
      # if it's better, take it
      pset[t,j]=pstar[j]
      lpset[t,j]=lpstar[j]
    } else { 
      # if it's worse, reject it (with some probability)
      pset[t,j]=pset[t-1,j]
      lpset[t,j]=lpset[t-1,j]
    }
  }
}

# plot it
for(j in 1:S){
par(mfrow=c(1,2))
# this is called a trace plot
plot(pset[,j],type="l",xlab="Iteration",ylab=paste("subject",j),ylim=c(0,1))
# this is the result of the MCMC, collapsed across time
hist(pset[,j],prob=TRUE,breaks=30,xlim=c(0,1))
}


maps=apply(pset,2,mean)
plot(maps,true$p,xlab="maps",ylab="true")
abline(0,1)


################################################################# posterior approximation, method 2: estimating alpha and beta

# first, select a prior (spend some time actually doing this!)

prior=list()
# somewhat informed
prior$alpha=c(20,1)
prior$beta=c(20,1)

# totally noninformative prior
# prior$alpha=c(.1,.1)
# prior$beta=c(.1,.1)


# inspect
hist(rgamma(1e5,20,1))

# wrap it up
log.dens.hyper=function(x,ps,prior){
  names(x) <- hpar.names
  pprior=dgamma(x["alpha"],prior$alpha[1],prior$alpha[2],log=TRUE) + dgamma(x["beta"],prior$beta[1],prior$beta[2],log=TRUE)
  pdens=dbeta(ps,x["alpha"],x["beta"],log=TRUE)
  out=sum(pdens)+pprior
  if(is.na(out))out=-Inf
  return(out)
}

####################################

n.iter=10000    # total number of iterations
pset=lpset=matrix(NA,n.iter,S)

phi=matrix(NA,n.iter,2)
hpar.names=c("alpha","beta")
colnames(phi) <- hpar.names
n.hyper.pars=length(hpar.names)

tune=.1
tune.hyper=3

# initialize
# informed starting points
#pset[1,]=sapply(data,function(x)mean(x))

# 'your guess is as good as mine' starting points
pset[1,]=rbeta(S,1,1)

# to initialize, you can either pick a value for alpha and beta and hope for the best, or simply use the likelihood
for(j in 1:S){
  # alpha=5
  # beta=5
  # lpset[1,j]=log.dens.post(pset[1,j],use.data=data[[j]],alpha=alpha,beta=beta)
  lpset[1,j]=log.dens.like(pset[1,j],use.data=data[[j]])
}


phi[1,]=c(5,5) # there are ways of picking alpha/beta based on initialized pset
log.dens.hyper(phi[1,],ps=pset[1,],prior=prior)

# loop
for(t in 2:n.iter){
  # first, update all p values based on the current alpha and beta.
  pstar=rnorm(S,pset[t-1,],tune) # symmetric transition kernel
  lpstar=numeric(S)
  for(j in 1:S){
    lpstar[j]=log.dens.post(pstar[j],use.data=data[[j]],alpha=phi[t-1,"alpha"],beta=phi[t-1,"beta"])
    a=exp(lpstar[j]-lpset[t-1,j]) # note that proposal density is in the numerator
    if(runif(1)<a){ 
      # if it's better, take it
      pset[t,j]=pstar[j]
      lpset[t,j]=lpstar[j]
    } else { 
      # if it's worse, reject it (with some probability)
      pset[t,j]=pset[t-1,j]
      lpset[t,j]=lpset[t-1,j]
    }
  }
  
  # now update the alpha and beta based on the (potentially) new ps
  phistar=rnorm(n.hyper.pars,phi[t-1,],rep(tune.hyper,n.hyper.pars))
  num=log.dens.hyper(phistar,ps=pset[t,],prior=prior)
  den=log.dens.hyper(phi[t-1,],ps=pset[t,],prior=prior)
  a=exp(num-den)
  if(runif(1)<a){ 
    # if it's better, take it
    phi[t,]=phistar
  } else { 
    # if it's worse, reject it (with some probability)
    phi[t,]=phi[t-1,]
  }
}

# plot it
for(j in 1:S){
  par(mfrow=c(1,2))
  # this is called a trace plot
  plot(pset[,j],type="l",xlab="Iteration",ylab=paste("subject",j),ylim=c(0,1))
  abline(h=true$p[j],col="red",lwd=3)
  # this is the result of the MCMC, collapsed across time
  hist(pset[,j],prob=TRUE,breaks=30,xlim=c(0,1))
  abline(v=true$p[j],col="red",lwd=3)
}

xs=seq(0,50,1)
ys.alpha=dgamma(xs,prior$alpha[1],prior$alpha[2])
ys.beta=dgamma(xs,prior$beta[1],prior$beta[2])

lims=c(0,30)
par(mfrow=c(2,2))
plot(phi[,1],type="l",xlab="Iteration",ylab=expression(alpha),ylim=lims)
abline(h=true$alpha,lty=1,lwd=3,col="red")
hist(phi[,1],prob=TRUE,breaks=30,xlim=lims,main="",xlab=expression(alpha))
abline(v=true$alpha,lty=1,lwd=3,col="red")
lines(xs,ys.alpha,lty=1,col="blue")

plot(phi[,2],type="l",xlab="Iteration",ylab=expression(beta),ylim=lims)
abline(h=true$beta,lty=1,lwd=3,col="red")
hist(phi[,2],prob=TRUE,breaks=30,xlim=lims,main="",xlab=expression(beta))
abline(v=true$beta,lty=1,lwd=3,col="red")
lines(xs,ys.beta,lty=1,col="blue")

par(mfrow=c(1,1))
maps=apply(pset,2,mean)
plot(maps,true$p,xlab="maps",ylab="true")
abline(0,1)
