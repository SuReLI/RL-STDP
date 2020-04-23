# %% Import Librairies

using Plots

# %% Parameter variables

M = 100          # Number of synapses per neurons
D = 1            # Maximal Conduction Delay (msec)
Ne = 800         # Excitatory neuron population size
Ni = 200         # Inhibitory neuron population size
N = Ne + Ni      # Total neuron population size
sm = 4           # Max synaptic strength (mV)

a = [(i<=Ne) ? 0.02 : 0.1 for i in 1:N]
d = [(i<=Ne) ? 8 : 2 for i in 1:N]

post = vcat(rand(1:N,Ne,M),rand(1:Ne,Ni,M));  # Creates the 100 random connections per neuron allowing E->I, E->E, I->E and not I->I (CONNEXION MATRIX)

s = vcat(ones(Ne,M),-ones(Ni,M))   # synaptic strength 1 for excitatory and -1 for inhibitory (SYNAPTIC WEIGHTS)
sd = zeros(N,M);                   # synaptic strength derivative, s = s+sd when dopamine is released (SYNAPTIC CHANGE IN WEIGTHS)


# %% Initailizing delays and STDP

delays = []
pre = []

for i in 1:N
    if i <= Ne
        del_temp = []
        for j in 1:D
            start = (M/D)*(j-1)+1
            fin = M/D*(j)
            append!(del_temp,[collect(start:fin)])
        end
    else
        del_temp = [[] for i in 1:D]
        del_temp[1] = collect(1:M)
    end
    global delays = vcat(delays,del_temp)
    append!(pre,[collect([index for index in findall(x->x==i,post) if s[index]>0])]) # find the pre excitatory neurons of neuron i
end

STDP = zeros(N,1001+D)        # all synaptic traces stored in this matrix
v = -65*ones(N)               # initial values of the membrane potentials (mV)
u = 0.2*v                     # initial values of the membrane recovery variable
firings = [-D 0];             # spike firing times

# %% new parameter initialization related to DA-STDP

T = 3600                     # total simulation time (sec)
DA = 0                       # dopamine level above baseline
rew = []                     # reward attribution times

n1 = 1                       # presynaptic targeted neuron for the simulation
syn = 1                      # targeted synapse
n2 = post[n1,syn]            # postsynaptic targeted neuron
s[n1,syn] = 0                # targeted synapse's weight initialized to 0

interval = 20                # tolerated spike time interval between n1 and n2 for reward attribution (msec)
n1f = [-100]                 # final spike time of n1 (msec)
n2f = []                     # final spike time of n2 (msec)

shist = zeros(1000*T, 2);    # recording for the plots (s-history)

# %% Main loop of the simulation

for sec in 0:(T-1)                               # 1 hour simulation time
    @time for t in 1:1000                              # 1 sec simulation time
        I=13*(rand(N).-0.5)
        fired = findall(x->x>=30,v)              # find the neurons that spiked
        v[fired] .= -65                          # reinitialize neurons that spiked to rest potential
        u[fired] = u[fired]+d[fired]
        if length(fired)!=0
            STDP[fired,t+D] .= 0.1               # incerment the appropriate synaptic trace
        end
        for k in fired
            pre_neurons_k = [pre[k][i][1] for i in 1:length(pre[k])]
            sd[pre[k]] = sd[pre[k]]  .+  STDP[pre_neurons_k,t]                 # increase the syn. der. by the (syn.trace)x1 (remember to RECHECK (t) vs (t+1))
        end
        global firings = Int64.(vcat(firings,hcat(t*ones(length(fired)),fired)))    # actualize the list of firing times with the corresponding spiking neuron
        last_ = length(firings[:,1])
        while firings[last_,1]>sec*1000+t-D
            del = Int64(delays[firings[last_,2]][sec*1000+t-firings[last_,1]+1])
            ind = post[firings[last_,2], del]
            I[ind] += s[firings[last_,2],del]
            sd[firings[last_,2],del] = sd[firings[last_,2],del] .-1.5*STDP[ind,t+D]
            last_ -= 1
        end
        global v=v+0.5.*((0.04.*v.+5).*v.+140-u+I)
        global v=v+0.5.*((0.04.*v.+5).*v.+140-u+I)
        global u=u+a.*(0.2*v-u)
        STDP[:,t+D+1] = 0.95*STDP[:,t+D]
        global DA = DA*0.995
        if t%10==0
            s[1:Ne,:] = max.(0,min.(sm,s[1:Ne,:]+(0.002+DA)*sd[1:Ne,:]))
            global sd = 0.99*sd
        end
        if n1 in fired
            append!(n1f,sec*1000+t)
        end
        if n2 in fired
            append!(n2f,sec*1000+t)
            if (sec*1000+t-last(n1f)<interval) && (last(n2f)>last(n1f))
                append!(rew,sec*1000+t+1000+rand(1:2000))
            end
        end
        if (sec*1000+t) in rew
            DA += 0.5
        end
        shist[1000*sec+t,:] = [s[n1,syn],sd[n1,syn]]
    end
    STDP[:,1:D+1]=STDP[:,1001:1001+D]
    ind = findall(x->x>1001-D,firings[:,1])
    global firings = Int64.(vcat([-D 0],hcat(firings[ind,1].-1000,firings[ind,2])))
    if sec%100==0
        print("\rsec = $sec")
    end
end

# %% Plot learning of the targeted synapse

gr()
x1 = 0.001.*collect(1:length(shist[:,1]))
y1 = shist[:,1]
x2 = x1
y2 = 10*shist[:,2]
fig = plot()
plot!(x1,y1,color="blue",label="synapse weight", legend = true)
plot!(x2,y2,color="green",label="synapse weight derivative", legend = true)
xlabel!("Time (sec)")
