'''
Copied from: https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py

Modified by: Maxlinn (https://github.com/Maxlinn/CHAIR-metric-standalone/tree/main)

1. adapt calculation of CHAIR-i and CHAIR-s for Python3, supports for both json and jsonl file input.
2. integrate synonyms.txt to make the script standalone.
3. remove machine-translation based metrics BLEU-n, CIDEr, ROGUE
4. add new metric Recall, which represents the node words(i.e. lemmas of objects) coverage overall.
5. add pickle cache mechanism to make it fast for repetitive evaluations.
'''

import os
import sys
import nltk
import json
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import argparse
from tqdm import tqdm
import pickle
from collections import defaultdict

# Ensure necessary NLTK data is downloaded
for pkg in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''


def combine_coco_captions(annotation_path):
    val_path = os.path.join(annotation_path, 'captions_val2014.json')
    train_path = os.path.join(annotation_path, 'captions_train2014.json')
    
    if os.path.exists(val_path) and os.path.exists(train_path):
        val_caps = json.load(open(val_path, 'r', encoding='utf-8'))
        train_caps = json.load(open(train_path, 'r', encoding='utf-8'))
        all_caps = {'info': train_caps.get('info', {}),
                    'licenses': train_caps.get('licenses', []),
                    'images': val_caps['images'] + train_caps['images'],
                    'annotations': val_caps['annotations'] + train_caps['annotations']}
        return all_caps
    elif os.path.exists(val_path):
        val_caps = json.load(open(val_path, 'r', encoding='utf-8'))
        return val_caps
    else:
        raise FileNotFoundError(f"Cannot find captions_val2014.json in {annotation_path}")


def combine_coco_instances(annotation_path):
    val_path = os.path.join(annotation_path, 'instances_val2014.json')
    train_path = os.path.join(annotation_path, 'instances_train2014.json')
    
    if os.path.exists(val_path) and os.path.exists(train_path):
        val_instances = json.load(open(val_path, 'r', encoding='utf-8'))
        train_instances = json.load(open(train_path, 'r', encoding='utf-8'))
        all_instances = {'info': train_instances.get('info', {}),
                         'licenses': train_instances.get('licenses', []),
                         'type': train_instances.get('type', ''),
                         'categories': train_instances['categories'],
                         'images': train_instances['images'] + val_instances['images'],
                         'annotations': val_instances['annotations'] + train_instances['annotations']}
        return all_instances
    elif os.path.exists(val_path):
        val_instances = json.load(open(val_path, 'r', encoding='utf-8'))
        return val_instances
    else:
        raise FileNotFoundError(f"Cannot find instances_val2014.json in {annotation_path}")


class CHAIR(object):
    def __init__(self, coco_path):
        self.imid_to_objects = defaultdict(list)
        self.coco_path = coco_path

        # read in synonyms
        synonyms = [s.strip().split(', ') for s in synonyms_txt.strip().splitlines() if s.strip()]
        self.mscoco_objects = []
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # common 'double words' in MSCOCO
        coco_double_words = [
            'motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 
            'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 
            'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 
            'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 
            'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 
            'home plate', 'train track'
        ]
        
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' % animal_word] = animal_word
            self.double_word_dict['adult %s' % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' % vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'
        
        if self.coco_path:
            self.get_annotations()

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):
        self.caps, self.eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
        assert len(self.caps) == len(self.eval_imids)

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def caption_to_words(self, caption):
        words = nltk.word_tokenize(caption.lower())
        tagged_sent = nltk.pos_tag(words)
        lemmas_sent = []
        wnl = WordNetLemmatizer()
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        words = lemmas_sent
    
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i) 
            double_word = ' '.join(words[i:i+2])
            if double_word in self.double_word_dict: 
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words
    
        if ('toilet' in words) and ('seat' in words):
            words = [word for word in words if word != 'seat']
    
        idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']

        id_to_name = {}
        for cat in coco_segments['categories']:
            id_to_name[cat['id']] = cat['name']

        for i, annotation in enumerate(segment_annotations):
            imid = int(annotation['image_id'])
            node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
            self.imid_to_objects[imid].append(node_word)

    def get_annotations_from_captions(self):
        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            imid = int(annotation['image_id'])
            _, node_words, _, _ = self.caption_to_words(annotation['caption'])
            self.imid_to_objects[imid].extend(node_words)

    def get_annotations(self):
        self.get_annotations_from_segments() 
        self.get_annotations_from_captions()
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def compute_chair(self, cap_file, image_id_key="image_id", caption_key="caption"):
        self._load_generated_captions_into_evaluator(cap_file, image_id_key, caption_key)
        
        imid_to_objects = self.imid_to_objects
        caps = self.caps
        eval_imids = self.eval_imids
 
        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        
        num_gt_objects = 0.
        num_recall_gt_objects = 0.
        len_caps = 0.

        output = {'sentences': []} 

        for i, (cap, imid) in enumerate(tqdm(zip(caps, eval_imids), total=len(caps), desc="Computing CHAIR")):
            imid = int(imid)
            raw_words, node_words, idxs, raw_tokens = self.caption_to_words(cap)
            gt_objects = imid_to_objects.get(imid, set())

            words = node_words
            cap_dict = {
                'image_id': imid,
                'caption': cap,
                'mscoco_hallucinated_words': [],
                'mscoco_gt_words': list(gt_objects),
                'mscoco_generated_words': list(words),
                'hallucination_idxs': [],
                'words': raw_tokens,
                'metrics': {
                    'CHAIRs': 0,
                    'CHAIRi': 0.,
                    'Recall': 0.,
                    'Len': 0.
                }
            }

            len_caps += len(raw_tokens)
            recall_gt_objects = set()
            for word, raw_word, idx in zip(words, raw_words, idxs):
                if word not in gt_objects:
                    cap_dict['mscoco_hallucinated_words'].append((raw_word, word))
                    cap_dict['hallucination_idxs'].append(idx)
                else:
                    recall_gt_objects.add(word)

            coco_word_count += len(words)
            hallucinated_word_count += len(cap_dict['mscoco_hallucinated_words'])

            if len(cap_dict['mscoco_hallucinated_words']) > 0:
                num_hallucinated_caps += 1.0
            num_caps += 1.0

            num_gt_objects += len(gt_objects)
            num_recall_gt_objects += len(recall_gt_objects)

            hallucinated = len(cap_dict['mscoco_hallucinated_words']) > 0
            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words']) / float(len(words)) if len(words) > 0 else 0.
            cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects) if len(gt_objects) > 0 else 0.
            cap_dict['metrics']['Len'] = len(raw_tokens)

            output['sentences'].append(cap_dict)

        chair_s = (num_hallucinated_caps / num_caps) if num_caps > 0 else 0.0
        chair_i = (hallucinated_word_count / coco_word_count) if coco_word_count > 0 else 0.0
        recall = (num_recall_gt_objects / num_gt_objects) if num_gt_objects > 0 else 0.0
        avg_len = (len_caps / num_caps) if num_caps > 0 else 0.0

        output['overall_metrics'] = {
            'CHAIRs': chair_s * 100.0,
            'CHAIRi': chair_i * 100.0,
            'Recall': recall * 100.0,
            'Caption_Length': avg_len,
        }

        return output


def load_generated_captions(cap_file, image_id_key: str, caption_key: str):
    ext = os.path.splitext(cap_file)[-1]
    if ext == '.json':
        with open(cap_file, 'r', encoding='utf-8') as f:
            caps = json.load(f)
    elif ext == '.jsonl':
        with open(cap_file, 'r', encoding='utf-8') as f:
            caps = [json.loads(s) for s in f if s.strip()]
    else:
        raise ValueError(f'Unsupported extension {ext} for cap_file: {cap_file}')

    imids = [int(obj[image_id_key]) for obj in caps]
    captions = [str(obj[caption_key]) for obj in caps]
    return captions, imids


def save_hallucinated_words(save_path, cap_dict):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(cap_dict, f, indent=2, ensure_ascii=False)


def print_metrics(hallucination_cap_dict):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    print("\n" + "=" * 45)
    print("           CHAIR EVALUATION METRICS          ")
    print("=" * 45)
    for k, v in sentence_metrics.items():
        k_str = str(k).ljust(18)
        unit = "%" if k in ["CHAIRs", "CHAIRi", "Recall"] else ("" if "total" in k.lower() else "words")
        val_str = f"{int(v)}" if "total" in k.lower() else f"{v:.2f}"
        print(f"  {k_str}: {val_str} {unit}".rstrip())
    print("=" * 45 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CHAIR Metric Standalone Evaluator")
    parser.add_argument("--cap_file", type=str, required=True,
                        help="Path towards json or jsonl saving image ids and captions.")
    parser.add_argument("--image_id_key", type=str, default="image_id",
                        help="Key storing image id.")
    parser.add_argument("--caption_key", type=str, default="caption",
                        help="Key storing generated caption.")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to pre-inited chair.pkl cache file.")
    parser.add_argument("--coco_path", type=str, default=None,
                        help="Directory with COCO annotation json files.")
    parser.add_argument("--save_path", type=str, default="",
                        help="Path to save detailed per-sentence evaluation json.")
    args = parser.parse_args()

    default_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chair.pkl")
    cache_path = args.cache or default_cache

    if os.path.exists(cache_path):
        print(f"[CHAIR] Loading evaluator from cache: {cache_path}")
        evaluator = pickle.load(open(cache_path, 'rb'))
    else:
        if not args.coco_path:
            raise FileNotFoundError(f"Cache file {cache_path} not found and --coco_path not provided.")
        print(f"[CHAIR] Building evaluator from COCO annotations at: {args.coco_path}")
        evaluator = CHAIR(args.coco_path)
        with open(cache_path, 'wb') as f:
            pickle.dump(evaluator, f)
        print(f"[CHAIR] Cached evaluator saved to: {cache_path}")

    res = evaluator.compute_chair(args.cap_file, args.image_id_key, args.caption_key)
    print_metrics(res)

    if args.save_path:
        save_hallucinated_words(args.save_path, res)
        print(f"[CHAIR] Detailed annotations saved to: {args.save_path}")
