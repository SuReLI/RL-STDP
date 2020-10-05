using PyCall
gym = pyimport("gym")


# %% swingup functions

function swingup_reset!(env::PyObject)
    env.reset()
    env.env.state += [0.0, 0.0, pi, 0.0]
    env.env.state[3] = (env.env.state[3]+pi)%2pi-pi
    return env.state
end

function swingup_step(env::PyObject, action::Int64, reward_shape::Bool = false)
    new_s, r , done_cart , _ = env.step(action)
    new_s[3] = mod(new_s[3]+pi, 2pi) -pi
    env.env.state = new_s
    if reward_shape
        if done_cart
            thresh = 12*pi/180
            rs_max = 0.1
            r = rs_max * (pi - abs(new_s[3]))/(pi - thresh)
        end
    end
    done = (abs(new_s[1]) > 2.4) | (abs(new_s[4]) > 4pi )
    return new_s, r, done, []
end

# %% tests
#
# env = gym.make("CartPole-v1")
# env.seed(0)
# swingup_reset!(env)
# @show env.state
# new_state, r ,done, _ = swingup_step(env, 1,true)
# @show r
# new_state, r ,done, _ = swingup_step(env, 0, true)
# @show r
# # for j in 1:100
# #     new_state, r ,done, _ = swingup_step(env, Int(((-1)^j+1)/2))
# #     @show done
# #     @show new_state
# # end
#
# env.close()
# env = nothing
