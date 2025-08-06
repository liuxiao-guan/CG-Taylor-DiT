def cache_init_step(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}
    cache['hidden'] = {}
    cache['firstblock_hidden']= {}
    

    for j in range(28):
        cache[-1][j] = {}
    for i in range(num_steps):
        cache[i]={}
        for j in range(28):
            cache[i][j] = {}

    cache_dic['cache']                = cache
    cache_dic['flops']                = 0.0
    cache_dic['interval']             = model_kwargs['interval']
    cache_dic['max_order']            = model_kwargs['max_order']
    cache_dic['test_FLOPs']           = model_kwargs['test_FLOPs']
    cache_dic['first_enhance']        = 2
    cache_dic['cache_counter']        = 0
    cache_dic['mode']                 = model_kwargs['mode'] # "Firstblock" # or firstblockpredict
    cache_dic['firstblock_max_order'] = model_kwargs['max_block_order']
    
    current = {}
    current['num_steps'] = num_steps
    current['block_activated_steps'] = [49]
    current['activated_steps'] = [49]
    return cache_dic, current
    