import os, subprocess

def TEOB_info(input_module,verbose=False):
    module = {}
    module['softlink']      = input_module.__file__
    module['name']          = module['softlink'].split('/')[-1]
    module['real_path']     = os.path.realpath(module['softlink'])
    
    teob_path = module['real_path'].replace(module['name'],'')
    teob_path = teob_path.replace('Python/','')
    module['teob_path'] = teob_path 
    module['commit']    = subprocess.Popen(['git', '--git-dir='+teob_path+'.git',\
                                            'rev-parse', 'HEAD'],stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    module['branch']    = subprocess.check_output(['git','--git-dir='+teob_path+'.git', \
                                            'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.STDOUT, text=True).strip()
    if verbose:
        for key,value in module.items():
            print(f'{key:10s} : {value}')
    return module 

if __name__ == '__main__':
    import EOBRun_module
    module = TEOB_info(EOBRun_module,verbose=True)
    #for key,value in module.items():
    #    print(f'{key:10s} : {value}')
