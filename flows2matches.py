import argparse
import os, numpy as np
from pathlib import Path
import torch, time
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Establish correspondences from optical flows')
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Database directory")
    parser.add_argument("--set-name", type=str, required=True,
                        help="Set name")
    args = parser.parse_args()

    return args


def load_flow(flow_dir, id_from, id_to):
    flow_file = os.path.join(flow_dir, "flow_" + id_from + "_to_" + id_to + ".npy")
    if os.path.exists(flow_file):
        flow = torch.from_numpy(np.load(flow_file))
        return flow
    
    flow_file = os.path.join(flow_dir, "flow_" + id_to + "_to_" + id_from + ".npy")
    flow = torch.from_numpy(np.load(flow_file))
    return -flow


def flow_to_matches(flow):
    hgt, wid = flow.shape[0], flow.shape[1]
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
    matches_y, matches_x = torch.meshgrid(torch.arange(hgt), torch.arange(wid), indexing='ij')
    matches_y = torch.round(matches_y + flow_y).type(torch.int)
    matches_x = torch.round(matches_x + flow_x).type(torch.int)
    return matches_x, matches_y

def is_valid_coord(x,y,wid,hgt):
    if x >= 0 and x < wid and y >=0 and y < hgt:
        return True
    return False

def is_valid_match(match, wid, hgt):
    x1, y1, x2, y2 = match[0], match[1], match[2], match[3]

    if is_valid_coord(x=x1, y=y1, wid=wid, hgt=hgt) and is_valid_coord(x=x2, y=y2, wid=wid, hgt=hgt):
        return True
    return False

def find_fund(matches):
    F, mask = cv2.findFundamentalMat(
        matches[:,:2].cpu().numpy(), matches[:,2:].cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
    )
    matches = matches[mask.ravel()==1, :]
    return matches, F

def main(args):
    base_dir = args.base_dir
    set_name = args.set_name
    flow_dir = os.path.join(base_dir, set_name, 'lwir_flow')
    os.makedirs(os.path.join(base_dir, set_name, "lwir_matches"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, set_name, "lwir_inliers"), exist_ok=True)

    hgt,wid = 256, 256
    train_list = os.path.join(base_dir, set_name + ".txt")
    with open(train_list) as f:
        contents = f.readlines()
        image_list = [i.strip() for i in contents]
        image_list = [i.replace('/lwir/', '/lwir_flow/') for i in image_list]
        ids = [Path(i).stem for i in image_list]
        
    n = len(ids)
    start_time = time.time()
    for k in range(n-1):
        id_from = ids[k]
        id_to = ids[k+1]
        flow = load_flow(flow_dir=flow_dir, id_from=id_from, id_to=id_to)

        mx, my = flow_to_matches(flow)
        matches = -torch.ones((hgt*wid,4)).type(torch.int)

        i = 0
        valid_ = []
        for y in range(hgt):
            for x in range(wid):
                # matches[i, :] = torch.tensor([mx[y,x], my[y,x], x,y]) # remember to check this order
                matches[i, :] = torch.tensor([x, y, mx[y,x], my[y,x]]) # remember to check this order
                if is_valid_match(matches[i, :], wid=wid, hgt=hgt):
                    valid_.append(i)
                i += 1

        matches = matches[valid_, :]
        inliers, _ = find_fund(matches)
        matches = matches.numpy()
        inliers = inliers.numpy()

        print(f"[{k+1}/{n}] # matches = {matches.shape[0]}, # inliers = {inliers.shape[0]}")
        if (k+1) % 100 == 1:
            print("\t ==> Elapsed time = %.2fs" % (time.time()-start_time))

        # save
        save_name = id_from + "_and_" + id_to
        np.save(os.path.join(base_dir, set_name, "lwir_matches", save_name + ".npy"), matches)
        np.save(os.path.join(base_dir, set_name, "lwir_inliers", save_name + ".npy"), inliers)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)