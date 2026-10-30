import os.path
from .regex import ImageFolder

class WriterZoo:

    @staticmethod
    def new(desc, **kwargs):
        return ImageFolder(desc['path'], regex=desc['regex'], **kwargs)

    @staticmethod
    def get(dataset, set, **kwargs):
        _all = WriterZoo.datasets
        d = _all[dataset]
        s = d['set'][set]

        s['path'] = os.path.join(d['basepath'], s['path'])
        return WriterZoo.new(s, **kwargs)

    datasets = {

        'icdar2017': {
            'basepath': '/data/rrossner/DAS2026/icdar',
            'set': {
                'test': {
                    'path': 'test_patches_coords',
                    'regex': {
                        'writer': r'^(\d+)-IMG_MAX_\d+',
                        'page':   r'^\d+-IMG_MAX_(\d+)'
                    }
                },
                'train': {
                    'path': 'train_patches',
                    'regex': {
                        'cluster': r'^(\d+)_',
                        'writer':  r'^\d+_(\d+)-',
                        'page':    r'^\d+_\d+-IMG_MAX_(\d+)'
                    }
                },
            }
        },

        'icdar2013': {
            'basepath': '/data/mpeer/resources',
            'set': {
                'test' :  {'path': 'icdar2013_test_sift_patches_binarized',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},

                'train' :  {'path': 'icdar2013_train_sift_patches_1000/',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}}                             
            }
        },
        
        'icdar2019': {
            'basepath': '/data/mpeer/resources',
            'set': {
                'test' :  {'path': 'wi_comp_19_test_patches',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},

                'train' :  {'path': 'wi_comp_19_validation_patches',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}},
            }
        },
        'cvl': {
            'basepath': '/data/rrossner/datasets/CVL',
            'set': {
                'train': {
                    'path':  'cvl_train_patches_bin',
                    # filename after extraction: 0001-2-cropped_42.png
                    'regex': {
                        'writer': r'(\d+)-\d+-cropped_\d+',
                        'page':   r'\d+-(\d+)-cropped_\d+',
                    }
                },
                'test': {
                    'path':  'cvl_test_patches_bin',
                    'regex': {
                        'writer': r'(\d+)-\d+-cropped_\d+',
                        'page':   r'\d+-(\d+)-cropped_\d+',
                    }
                },
            }
        },
        'cvl_cluster_32': {
            'basepath': '/data/rrossner/datasets/CVL',
            'set': {
                'train': {
                    'path':  'cvl_train_patches_c32',
                    # filename after extraction: 0001-2-cropped_42.png
                    'regex': {
                        'cluster':r'(\d+)',
                        'writer': r'\d+_(\d+)-\d+-cropped',
                        'page':   r'\d+_\d+-(\d+)-cropped',
                    }
                },
                'test': {
                    'path':  'cvl_test_patches_bin',
                    'regex': {
                        'writer': r'(\d+)-\d+-cropped_\d+',
                        'page':   r'\d+-(\d+)-cropped_\d+',
                    }
                },
            }
        },
        # MASLOTS Sanskrit hands (Patrick-data), patches: 0_0012_345.png = writer_page_count
        # writer = hand id, see /data/rrossner/datasets/Patrick-data/labels.csv
        'maslots_skt': {
            'basepath': '/data/rrossner/datasets',
            'set': {
                'test': {
                    'path':  'Patrick-data_patches',
                    'regex': {'writer': r'^(\d+)_', 'page': r'^\d+_(\d+)_'}
                },
            }
        },

        'maslot': {
            'basepath': '/data/rrossner/datasets',
            'set': {
                # filename after extraction: 12_4853_48401_7.png = cluster_manuscript(recnum)_page(imgid)_count
                'train': {
                    'path':  'MaSLoT_Sift_Patches_Color',
                    'regex': {
                        'cluster': r'^(\d+)_',
                        'writer':  r'^\d+_(\d+)_\d+_\d+',
                        'page':    r'^\d+_\d+_(\d+)_\d+',
                    }
                },
                'test': {
                    'path':  'MaSLoT_Sift_Patches_Color',
                    'regex': {
                        'writer': r'^\d+_(\d+)_\d+_\d+',
                        'page':   r'^\d+_\d+_(\d+)_\d+',
                    }
                },
            }
        },
        'iam': {
            'basepath': '/data/rrossner/datasets/IAM',
            'set': {
                'train': {
                    'path':  'iam_train_patches',
                    # filename after extraction: 001_0001_0.png
                    'regex': {
                        'writer': r'^(\d+)_',
                        'page':   r'^\d+_(\d+)_',
                    }
                },
                'test': {
                    'path': 'iam_test_patches',
                    'regex': {
                        'writer': r'^(\d+)_',
                        'page':   r'^\d+_(\d+)_',
                    }
                },
            }
        },
        'iam_threshold': {
            'basepath': '/data/rrossner/datasets/IAM',
            'set': {
                'train': {
                    'path':  'train_threshold_patches',
                    # filename after extraction: 0_a01-007u_256.png
                    'regex': {
                        'cluster': r'^(\d+)',
                        'writer':  r'^\d+_([a-zA-Z0-9]+)-',
                        'page':    r'^\d+_[a-zA-Z0-9]+-([a-zA-Z0-9]+)_',
                    }
                },
                'test': {
                    'path': 'test_threshold_patches',
                    'regex': {
                        'writer': r'^([a-zA-Z0-9]+)-',
                        'page':   r'^[a-zA-Z0-9]+-([a-zA-Z0-9]+)_',
                    }
                },
            }
        },
    }