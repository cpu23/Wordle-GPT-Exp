"""Five folds x three seeds: immutable hard SFT versus soft B and C at T=0.25."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from cross_validation import aggregate_seed_summaries, combine_fold_predictions, compare_paired_predictions, load_mode
from evaluate_v2 import evaluate_checkpoint, load_v2_model
from soft_evaluate import evaluate_soft_model
from soft_teacher import TeacherDataset, build_teacher_dataset
from train_soft_distillation import deterministic_execution, file_hash, split_indices, stop_signals, train_configuration, write_json
from train_v2 import load_v2_split
from wordle import load_words


def publish_aggregates(args, mode):
    combined = {}
    for variant in ('hard-sft', 'B', 'C'):
        seeds = []
        for seed in mode.model_seeds:
            paths = [args.output_dir / variant / f'seed-{seed}' / f'fold-{fold.run}' / 'held-out-games.json' for fold in mode.runs]
            if not all(path.exists() for path in paths):
                continue
            result = combine_fold_predictions(mode, paths)
            if result['games'] != 719:
                raise ValueError('each model/seed must cover exactly719 held-out secrets')
            write_json(args.output_dir / variant / f'seed-{seed}' / 'combined.json', result)
            seeds.append(result)
            combined[(variant,seed)] = result
        if len(seeds) == 3:
            write_json(args.output_dir / variant / 'aggregate.json', aggregate_seed_summaries(seeds))
    for variant in ('B', 'C'):
        paired = []
        for seed in mode.model_seeds:
            if ('hard-sft',seed) in combined and (variant,seed) in combined:
                comparison = compare_paired_predictions(combined[('hard-sft',seed)], combined[(variant,seed)])
                paired.append({'seed':seed, **comparison})
        if paired:
            write_json(args.output_dir / variant / 'paired.json', paired)
    complete = len(combined) == 9
    write_json(args.output_dir / 'progress.json', {'complete':complete, 'completed_model_seeds':
               [f'{variant}/seed-{seed}' for variant,seed in combined]})
    if complete:
        write_json(args.output_dir / 'benchmark-complete.json', {
            'decode':'word-argmax', 'temperature':.25, 'secrets_per_seed':719,
            'results': {variant:json.loads((args.output_dir/variant/'aggregate.json').read_text()) for variant in ('hard-sft','B','C')},
            'test_gameplay_evaluated':True})


def run(args):
    deterministic_execution()
    torch.set_num_threads(4)
    mode = load_mode(args.mode)
    words = tuple(load_words())
    if len(mode.runs) != 5 or tuple(mode.model_seeds) != (0,1,2) or len(words) != 719:
        raise ValueError('requires existing five folds, seeds0/1/2 and719 words')
    splits = json.loads(args.mode.read_text())['runs']
    cells = []
    for seed in mode.model_seeds:
        for fold in mode.runs:
            original = args.baseline_root / f'seed-{seed}' / f'fold-{fold.run}' / '7.2m'
            hard = original / 'checkpoints/best.pt'
            mechanics = original / 'mechanics/checkpoints/best.pt'
            cells.append({'seed':seed, 'fold':fold.run, 'hard':str(hard), 'hard_sha256':file_hash(hard),
                          'mechanics':str(mechanics), 'mechanics_sha256':file_hash(mechanics)})
    manifest = {'version':1, 'decode':'word-argmax', 'temperature':.25,
                'B_learning_rate':3e-4, 'C_learning_rate':1e-5, 'C_retry_learning_rate':3e-6,
                'C_retry_rule':'validation wins at step250 at least5 below initialization; never test-based',
                'mode':str(args.mode), 'mode_sha256':file_hash(args.mode), 'splits':splits,
                'source':str(args.source), 'source_sha256':file_hash(args.source),
                'teacher_dir':str(args.teacher_dir), 'baseline_cells':cells,
                'max_epochs':args.max_epochs, 'patience':args.patience,
                'microbatch_states':args.microbatch_states, 'effective_batch_states':128,
                'eval_batch_states':args.eval_batch_states, 'panel_size':args.panel_size,
                'selection':'validation only, word-MAP wins/attempts/guesses/regret/KL',
                'data_dir':str(args.data_dir), 'words':list(words)}
    manifest_path = args.output_dir / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('benchmark manifest changed; use a fresh output directory')
    write_json(manifest_path, manifest)
    with stop_signals() as stop:
        # Complete all15 hard baselines before any new training.
        for cell in cells:
            if stop['requested']:
                raise SystemExit(130)
            fold = mode.runs[cell['fold']-1]
            output = args.output_dir / 'hard-sft' / f"seed-{cell['seed']}" / f"fold-{cell['fold']}" / 'held-out-games.json'
            if not output.exists():
                report = asdict(evaluate_checkpoint(cell['hard'], fold.test, words, device=args.device, decode='word-argmax'))
                report['checkpoint_sha256'] = cell['hard_sha256']
                write_json(output, report)
                print(json.dumps({'event':'hard baseline complete','seed':cell['seed'],'fold':cell['fold'],'wins':report['wins']}),flush=True)
        publish_aggregates(args,mode)
        if args.baseline_only:
            return
        if stop['requested']:
            raise SystemExit(130)
        if not (args.teacher_dir / 'manifest.json').exists():
            build_teacher_dataset(args.teacher_dir, source=args.source, workers=args.workers, seed=20260924, expected_count=1_000_000)
        teacher_manifest_path = args.teacher_dir / 'manifest.json'
        teacher_manifest = json.loads(teacher_manifest_path.read_text())
        if (teacher_manifest['source']['sha256'] != manifest['source_sha256']
                or teacher_manifest['words'] != list(words)
                or teacher_manifest['rows'] != 1_000_000
                or teacher_manifest['seed'] != 20260924):
            raise ValueError('teacher cache does not match the frozen CV corpus/dictionary/seed')
        teacher_identity = {'manifest_sha256':file_hash(teacher_manifest_path)}
        identity_path = args.output_dir / 'teacher-identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != teacher_identity:
            raise ValueError('teacher cache changed during benchmark')
        for name, description in teacher_manifest['arrays'].items():
            if file_hash(args.teacher_dir / f'{name}.npy') != description['sha256']:
                raise ValueError(f'teacher array checksum mismatch: {name}')
        write_json(identity_path,teacher_identity)
        dataset = TeacherDataset(args.teacher_dir)
        for cell in cells:
            if stop['requested']:
                raise SystemExit(130)
            seed,fold = cell['seed'],mode.runs[cell['fold']-1]
            split = splits[fold.run-1]
            mechanics_dir = args.data_dir / f'fold-{fold.run}' / 'mechanics'
            mechanics_train = load_v2_split(mechanics_dir,'train',example_type='mechanics')
            mechanics_validation = load_v2_split(mechanics_dir,'validation',example_type='mechanics')
            val_indices = split_indices(dataset,fold.validation)
            rng = np.random.default_rng(seed+700_001)
            panel = np.sort(rng.choice(val_indices,min(args.panel_size,len(val_indices)),replace=False))
            for variant in ('B','C'):
                if stop['requested']:
                    raise SystemExit(130)
                output = args.output_dir / variant / f'seed-{seed}' / f'fold-{fold.run}'
                training_args = SimpleNamespace(seed=seed,device=args.device,output_dir=output,
                    microbatch_states=args.microbatch_states,eval_batch_states=args.eval_batch_states,
                    teacher_dir=args.teacher_dir,mode=args.mode,max_epochs=args.max_epochs,patience=args.patience,
                    smoke_steps=0,resume=args.resume,save_every=args.save_every,stop_after_updates=None)
                initialization = Path(cell['mechanics'] if variant=='B' else cell['hard'])
                lr = 3e-4 if variant=='B' else 1e-5
                result = train_configuration(training_args,dataset,split,panel,mechanics_train,mechanics_validation,
                                             variant,.25,initialization,lr,_stop=stop)
                if variant=='C' and result['rapid_destructive_drift']:
                    result = train_configuration(training_args,dataset,split,panel,mechanics_train,mechanics_validation,
                                                 variant,.25,initialization,3e-6,_stop=stop)
                if stop['requested']:
                    raise SystemExit(130)
                path = output / 'held-out-games.json'
                if not path.exists():
                    # No test gameplay until training/validation-only selection has finished.
                    test_indices = split_indices(dataset,fold.test)
                    test_rng = np.random.default_rng(seed+900_001)
                    test_panel = np.sort(test_rng.choice(test_indices,min(args.panel_size,len(test_indices)),replace=False))
                    model = load_v2_model(result['checkpoint'],args.device)
                    report = evaluate_soft_model(model,dataset,test_panel,.25,fold.test,words,mechanics_validation,
                                                 args.device,batch_size=args.eval_batch_states,distribution_examples=24)
                    report.update(checkpoint=result['checkpoint'],checkpoint_sha256=file_hash(result['checkpoint']),phase='held-out')
                    write_json(output/'held-out-evaluation.json',report)
                    write_json(path,report['gameplay'])
                    del model
                    torch.cuda.empty_cache()
                    print(json.dumps({'event':'soft heldout complete','variant':variant,'seed':seed,'fold':fold.run,
                                      'wins':report['gameplay']['wins']}),flush=True)
                publish_aggregates(args,mode)
            assert file_hash(cell['hard'])==cell['hard_sha256']
            assert file_hash(cell['mechanics'])==cell['mechanics_sha256']
            del mechanics_train,mechanics_validation


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('runs/soft-distillation-cv5-word-argmax'))
    p.add_argument('--teacher-dir',type=Path,default=Path('data/soft-teacher-cv1m'))
    p.add_argument('--source',type=Path,default=Path('data/wordle-v2-diverse-1m-cv/examples.jsonl.gz'))
    p.add_argument('--mode',type=Path,default=Path('data/wordle-cv5.json'))
    p.add_argument('--baseline-root',type=Path,default=Path('runs/scaling-cv5-1m'))
    p.add_argument('--data-dir',type=Path,default=Path('data/wordle-cv5-1m'))
    p.add_argument('--microbatch-states',type=int,default=128)
    p.add_argument('--eval-batch-states',type=int,default=16)
    p.add_argument('--panel-size',type=int,default=512)
    p.add_argument('--max-epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=4)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--baseline-only',action='store_true')
    return p


if __name__=='__main__':
    run(parser().parse_args())
