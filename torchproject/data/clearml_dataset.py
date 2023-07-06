import numpy as np
from clearml import Task
from clearml import Dataset
from typing import Optional, List, Union
import re
try:
    # unneccessary libs
    import matplotlib.pyplot as plt
except ImportError:
    pass

PROJECT_NAME = 'Obj_Detect'
EXAMPLES_DIR = '/mnt/raid10/datasets/clearml_examples'
MANIFEST_EXT = '.csv'
# where example files are stored


def parse_manifest_path(manifest_path:str) -> dict:
    """
    Args:
        manifest_path (str): /path/to/manifest.v{X}.{Y}-{postfix}
    Raises:
        ValueError: "Failed to extract version info 
                    from '{manifest_path}'"
    Returns:
        dict: {
            'basepath': /path/to/manifest',
            'major_version': X (int),
            'minor_version': Y (int),
            'postfix': Union[str, None]
        }
    """
    full_pattern = r'.*v\d+[.]\d+([-]\w+)?' + MANIFEST_EXT
    version_pattern = r'v\d+[.]\d+([-]\w+)?'
    if not re.fullmatch(full_pattern, manifest_path):
        raise ValueError("Failed to extract version info from "\
                         f"'{manifest_path}'")
    match_ = re.search(version_pattern, manifest_path)
    version_info = match_[0][1: ]  # '{X}.{Y}-{postfix}
    xy = version_info.split('-')[0]
    x, y = int(xy.split('.')[0]), int(xy.split('.')[1])
    try:
        postfix = version_info.split('-')[1]
    except IndexError:
        postfix = None
    out = {
        "major_version": x,
        "minor_version": y,
        "postfix": postfix
    }
    return out


def get_version_from_path(manifest_path:str) -> str:
    """
    Extract version info from manifest path
    Args:
        manifest_path (str): /path/to/file.v{X}.{Y}-{postfix}
    Raises:
        ValueError: "Failed to extract version info 
                    from '{manifest_path}'"
    Returns:
        str: '{X}.{Y}-{postfix}'
    """
    parsed = parse_manifest_path(manifest_path)
    x, y = parsed["major_version"], parsed["minor_version"]
    version = f"{x}.{y}"
    postfix = parsed["postfix"]
    if postfix:
        version += f"-{postfix}" 
    return version


def get_pie_figure(fracs:List[int], names:List[str]):
    """
    Plot pie figure (SVG filter pie)
    Args:
        fracs (list[int]): list of counts (int)
        names (list[str]):
    Returns:
        plt.figure
    """
    # validate input
    if len(fracs) != len(names):
        raise ValueError("Mismatch number of 'fracs' and 'names'")

    # to int persents
    s = sum(fracs)
    fracs = [int(fr / s * 100) for fr in fracs]
    assert 100 - len(fracs) <= sum(fracs) <= 100
    while sum(fracs) != 100:
        fracs[0] += 1
    
    # make a square figure and axes
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    explode = [0] * len(fracs)

    pies = ax.pie(fracs, explode=explode, labels=names, 
                  autopct='%1.1f%%')

    for w in pies[0]:
        # set the id with the label.
        w.set_gid(w.get_label())
        # disale edge
        w.set_edgecolor("none")

    return fig


class ClearmlDataset:
    """ Don't forget to do commit after uploading 
    plots, files, tables, etc
    """
    __project_name = 'Datasets/' + PROJECT_NAME
    __dataset_name = PROJECT_NAME.lower()

    def __init__(self, dataset:Dataset):
        """ Don't use this directly. User 'create' or 'get' instead """
        self.dataset = dataset
        self.logger = dataset.get_logger()
        self.id = dataset.id
        self.version = dataset._dataset_version

    @classmethod
    def create(cls, manifest:str,
               version:str=None,
               previous:Optional[List[str]] = None,
               description:Optional[str]=None):
        """
        Create new dataset with given version
        Args:
            manifest (str): /path/to/manifest.v{X}.{Y}-{postfix}.csv
            version (str): version in format like '1.2-test' 
                           or '1.0-mnist'
                           If None, version will be automatically parsed
                           from manifest_path
            previous (list[str]): parent IDs
            description (str): may refer to transformation 
                for example 'balancing'
        Returns:
            ClearmlDataset
        """
        if previous and not isinstance(previous, list):
            raise TypeError("'previous' must be list of IDs")
        if not version:
            # get version from manifest path
            version = get_version_from_path(manifest)

        # to connect git info task needs to be defined
        Task.init(project_name=cls.__project_name, 
                  task_name=cls.__dataset_name, 
                  task_type=Task.TaskTypes.data_processing,
                  output_uri=EXAMPLES_DIR,
                  auto_connect_frameworks=False,
                  auto_resource_monitoring=False, 
                  auto_connect_streams=False)
        dataset = Dataset.create(
                    parent_datasets=previous,
                    dataset_version=version,
                    description=description,
                    use_current_task=True)
        dataset._task.set_user_properties(manifest=manifest)
        return cls(dataset)

    @classmethod
    def get(cls, version:Optional[str]="latest", id:Optional[str]=None):
        """
        Get dataset with given version or given ID
        Args:
            version (str): version. If "latest" get newest version
            id (str): ID
            NOTE: use either 'version' or 'id'
        Returns:
            ClearmlDataset
        """
        if bool(version) == bool(id):
            raise ValueError("Define either 'version' or 'id'")
        
        if version == "latest":
            dataset = Dataset.list_datasets(cls.__project_name, 
                                            cls.__dataset_name)[-1]
            return cls(dataset)
        
        dataset = Dataset.get(dataset_id=id, 
                              dataset_version=version, 
                              dataset_project=cls.__project_name,
                              dataset_name=cls.__dataset_name,
                              only_completed=True)
        return cls(dataset)
    
    @classmethod
    def get_all(cls):
        """
        Get all datasets in given project
        Returns:
            list[str]: list of Dataset ID
        """
        datasets = Dataset.list_datasets(cls.__project_name, 
                                         cls.__dataset_name)
        ids = [ds["id"] for ds in datasets]
        return ids
    
    def add_metadata(self, datasize:int, **kwargs):
        """
        Metadata are visualized in 
        ClearML Task -> configuration -> properties
        It may refer to transformation info, like clean method 
        Args:
            datasize (int): number of examples - neccessary param
            key1: value
            key2: {"value": value, "description": "example"}
        """
        self.dataset._task.set_user_properties(datasize=datasize, **kwargs)

    def get_metadata(self, key:Optional[str]=None):
        """
        Args:
            key (str, None): if defined return only value of this key
                Otherwise return all metadata (dict)
        """
        metadata = self.dataset._task.get_user_properties()
        # drop unneccessary fields
        for k, data in metadata.items():
            metadata[k] = data["value"]
        if not key:
            return metadata
        return metadata[key]
    
    def get_manifest(self) -> str:
        manifest = self.get_manifest("manifest")
        return manifest

    def add_text(self, text:str, print=False):
        self.logger.report_text(text, print_console=print)

    def add_histogram(self, data, 
                      name:str='Histogram', 
                      series:str='data', 
                      xtitle:str='x',
                      ytitle:str='y'):
        """
        Args:
            data (np.ndarray): 1-D array
            name (str): name of histogram
            series (str): name of series
            NOTE: multiple histograms with same 'name'
            are plotted together with different 'series'
        """
        self.logger.report_histogram(name,
                                     series,
                                     values=data,
                                     xaxis=xtitle,
                                     yaxis=ytitle)

    def add_pd_table(self, data,
                     name:str='Table', 
                     series:str='data',
                     extra_layout=None):
        """
        Add table from pandas DataFrame
        Args:
            data (pd.DataFrame): data table
            name (str): name of table
            series (str): name of series
            extra_layout (str): additional config for table
                See https://plotly.com/javascript/reference/layout/
        """
        self.logger.report_table(
                title=name,
                series=series,
                table_plot=data,
                extra_layout=extra_layout
        )

    def add_csv_table(self, csv_path:str,
                     name:str='Table', 
                     series:str='data',
                     extra_layout=None):
        """
        Add table from CSV file
        Args:
            csv_path (str): /path/to/table.csv
            name (str): name of table
            series (str): name of series
            extra_layout (str): additional config for table
                See https://plotly.com/javascript/reference/layout/
        """
        self.logger.report_table(
                title=name,
                series=series,
                csv=csv_path,
                extra_layout=extra_layout
        )

    def add_figure(self, figure,
                   name:str='Figure',
                   series:str='data'
                   ):
        self.logger.report_matplotlib_figure(
            title=name,
            series=series,
            figure=figure,
            report_image=False,
            report_interactive=True
        )

    def add_example(self, fpath:str, 
                    name:Optional[str]='example'):
        """
        Add some media file as an example. It can be viewed in
        Task information -> Debug samples
        NOTE: Don't add large files
        Args:
            fpath (str): /path/to/file
        """
        self.logger.report_media(title='examples', 
                                 series=name,
                                 local_path=fpath,
                                 stream=None,
                                 delete_after_upload=False)

    def commit(self):
        self.dataset.finalize()



if __name__ == '__main__':
    # I. Create new dataset
    
    # define parents
    previous = None  # no parents
    # previous = ['48b8d22437624e519f6a177eddcfcd36']  # one parent
    # previous = ['8ccc9f696ebb4ea1b1b5f79adf200851',  # two parents
    #             'ada9f6d0fe0b4ee880b976988a09fdae']
    
    manifest = '/mnt/raid10/datasets/obj_detect/manifests/v1.2-valid.csv'
    version = None  # parsed from manifest - 1.2-valid
    # version = '1.4'
    dataset = ClearmlDataset.create(manifest=manifest,
                                    version=version, 
                                    previous=previous,
                                    description='Add more data'
    )

    # Histogram
    class_counts = [5545, 6987, 9564]
    hist = np.array(class_counts, dtype=np.uint16)
    dataset.add_histogram(hist, 'Class balance', 
                          xtitle='classes', 
                          ytitle='count')
    # Plot Pie
    data_size = sum(class_counts)
    fracs = [cls_count / data_size for cls_count in class_counts]
    fracs = [round(f * 100) for f in fracs]
    print(fracs)
    fig = get_pie_figure(fracs, ['pistol', 'rifle', 'knife'])
    dataset.add_figure(fig)

    # add metainfo
    dataset.add_metadata(
        datasize=26000,
        method="upsampling",
        alpha={"value": 0.45, "description": "balance factor"},
    )

    # # add table
    dataset.add_csv_table('/mnt/nvme/vovik/tests/dataset/examples/table.csv', 
                          name='Classes')

    # add example content
    examples = [
        '/mnt/nvme/vovik/tests/dataset/examples/1_target.jpg',
        '/mnt/nvme/vovik/tests/dataset/examples/8_target.jpg',
        '/mnt/nvme/vovik/tests/dataset/examples/4_target.jpg',
        '/mnt/nvme/vovik/tests/dataset/examples/9_target.jpg',
    ]
    for i, example in enumerate(examples):
        dataset.add_example(example, str(i))

    dataset.commit()


    # # II. Get dataset
    # dataset = ClearmlDataset.get(version='latest')
    # manifest = dataset.get_metadata('manifest')
    # print(dataset.id)
    # print(manifest)
