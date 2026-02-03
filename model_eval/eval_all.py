import numpy as np
import json
import argparse
import os

ds_name = {
    "travel_plan_gen":"/path/to/uni_bench/map/generation/map_gen.json",
    "travel_plan_edit":"/path/to/uni_bench/map/editing/map_editing_prompts.json",
    "process_plan_gen": "/path/to/uni_bench/diagram/generation/diagram_gen.json",
    "process_plan_edit":"/path/to/uni_bench/diagram/editing/diagram_edit.json",
    "ui_gen": "/path/to/uni_bench/UI/generation/ui_generation.json",
    "ui_gen_diff": "/path/to/uni_bench/UI/generation/ui_generation_diff.json",
    'ui_edit': "/path/to/uni_bench/UI/editing/ui_edit.json",
    "random_style": "/path/to/uni_bench/random_style.json"
}


def eval(args, generation_hyper=None):
    with open(ds_name[args.data_name], 'r') as fp:
        data = json.load(fp)
    fp.close()
    os.makedirs(f'results/{args.model_name}/{args.data_name}/', exist_ok=True)
    if "qwen" in args.model_name:
        from qwen_image import eval_qwen_image
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            content = eval_qwen_image(img, line['prompt'], args.model_name)
            if len(content) == 0:
                continue
            elif len(content) == 1:
                res_img_link = content[0]
                os.system(f"curl -L -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36' -o results/{args.model_name}/{args.data_name}/{i}.png '{res_img_link}'")
                print(f"Successfully Generate [{i+1}]")
                data[i]['id'] = i+1
                data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'

    elif "bagel" in args.model_name:
        from bagel import eval_bagel
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            if 'think' in args.model_name:
                think = True
            else:
                think = False
            content = eval_bagel(img, line['prompt'], edit=edit, think=think)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'

    elif "janus4o" in args.model_name: #environment use bagel
        from janus_4o import eval_janus4o
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            
            content = eval_janus4o(img, line['prompt'], edit=edit)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'
    
    elif "hidream" in args.model_name: #environment use bagel
        from hidream import eval_hidream  #accelerate start
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            
            content = eval_hidream(img, line['prompt'], edit=edit)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'
    
    elif "gpt" in args.model_name: #environment use bagel
        from gpt_gen import eval_gpt_gen
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            try:
                content = eval_gpt_gen(img, line['prompt'], edit=edit)
                content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
                print(f"Successfully Generate [{i+1}]")
                data[i]['id'] = i+1
                data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'
            except:
                pass

    elif "flux1_dev" in args.model_name: #environment use bagel
        from flux1_dev import eval_flux1_dev
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False

            content = eval_flux1_dev(img, line['prompt'], edit=edit)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'

    elif "anyedit" in args.model_name:
        #Edit only
        from anyedit import eval_anyedit
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False

            content = eval_anyedit(img, line['prompt'])
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'
    
    elif "step1x" in args.model_name:
        #Edit only
        from step1x import eval_step1x
        for i, line in enumerate(data):
            if 'think' in args.model_name:
                think = True
            else:
                think = False
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False

            content = eval_step1x(img, line['prompt'], think=think)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'

    elif "ultraedit" in args.model_name:
        #Edit only
        from ultraedit import eval_ultraedit
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False

            content = eval_ultraedit(img, line['prompt'])
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'

    elif "seedream" in args.model_name:
        from seedream import eval_seedream
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            try:
                content = eval_seedream(img, line['prompt'], edit=edit)
                content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
                print(f"Successfully Generate [{i+1}]")
                data[i]['id'] = i+1
                data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'
            except:
                pass

    elif "omnigen" in args.model_name: #env bagel
        from omni2 import eval_omni2
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            
            content = eval_omni2(img, line['prompt'], edit=edit)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'
    
    elif "sd3.5" in args.model_name: #env bagel
        from sd import eval_sd35
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            
            content = eval_sd35(img, line['prompt'], edit=edit)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'
    
    elif "nextstep" in args.model_name: #env bagel
        from nextstep import eval_nextstep
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            
            content = eval_nextstep(img, line['prompt'], edit=edit)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'
    
    elif "ovis" in args.model_name: #env bagel
        from ovis import eval_ovis
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            
            content = eval_ovis(img, line['prompt'], edit=edit)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'

    elif "unipic" in args.model_name: #env bagel
        from unipic2 import eval_unipic
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            if img is None or os.path.exists(img):
                pass 
            else:
                img = img.replace('png', 'jpg')
            if 'edit' in args.data_name:
                edit = True 
            else:
                edit = False
            
            content = eval_unipic(img, line['prompt'], edit=edit)
            content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
            print(f"Successfully Generate [{i+1}]")
            data[i]['id'] = i+1
            data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'

    elif "wan" in args.model_name:
        from qwen_image import eval_wan
        for i, line in enumerate(data):
            if 'image_path' in line:
                img = line['image_path']
            else:
                img = None
            try:
                content = eval_wan(img, line['prompt'], args.model_name)
                content.save(f'results/{args.model_name}/{args.data_name}/{i}.png')
                #if len(content) == 0:
                    #continue
                #elif len(content) == 1:
                    #res_img_link = content[0]
                    #os.system(f"curl -L -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36' -o results/{args.model_name}/{args.data_name}/{i}.png '{res_img_link}'")
                print(f"Successfully Generate [{i+1}]")
                data[i]['id'] = i+1
                data[i]['res_path'] = f'results/{args.model_name}/{args.data_name}/{i}.png'
            except:
                data[i]['id'] = i+1
    else:
        raise ValueError('No valuable Model Name Provided!')

    with open(f'results/{args.data_name}_{args.model_name}.json', 'w') as writer:
        json.dump(data, writer, ensure_ascii=False, indent=4)
    
    writer.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a simple argparse example")

    parser.add_argument('--data_name', type=str, help="evaluation_data_name", default="")
    parser.add_argument('--model_name', type=str, help="evaluation_model_name", default="")
    parser.add_argument('--verbose', action='store_true', help="Increase output verbosity")

    args = parser.parse_args()

    eval(args)
