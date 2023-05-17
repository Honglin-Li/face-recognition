import os
import tempfile
from src.models.approximate_k_nearest_neighbors import ApproximateKNearestNeighbors
from src.models.face_recognition import FaceRecognition
from src.postprocessing.graph_postprocessing import extract_scenes


class Hunter(object):
    """ Class to use the entity linking in other projects and on the website. """

    def __init__(self, url: str = None):
        """
        Args:
            url (str): URL of the video on YouTube.
        """
        self.url = url
        self.identifier = self.url.split('=')[1]
        self.path_to_video = None
        self.face_detection = None

    def fit(self,
            thumbnail_list=None,
            thumbnails_path='data/thumbnails/thumbnails',
            img_width=500,
            encoder_name: str ='Dlib',
            labels_path='data/embeddings/labels.pickle',
            embeddings_path='data/embeddings/embeddings.pickle'
            ):
        """ Creates the embeddings for a dictionary of thumbnails.

        Args:
            thumbnail_list (list): list of thumbnails to load.
            thumbnails_path (str): Path to the directory containing the thumbnails.
            img_width (int): Size to which the thumbnails should be resized.
            encoder_name (str): Specifies the method to create embeddings of faces in an image.
            labels_path (str): Path where the label-information should be saved.
            embeddings_path (str): Path where the embeddings should be saved.

        Returns:
            self
        """
        self.face_detection = FaceRecognition(
            thumbnail_list,
            thumbnails_path,
            img_width,
            encoder_name,
            labels_path,
            embeddings_path
        )
        return self

    def recognize(self,
                  algorithm='appr',
                  method='hnsw',
                  space='cosinesimil',
                  distance_threshold=0.4,
                  index_path='data/embeddings/index.bin',
                  k=1,
                  recognize_by: str = 'second'
                  ) -> list:
        """ Get a list of entities that could be recognized in the video.

        Args:
            algorithm (str): Algorithm to use for the similarity-calculation. Should be '1nn' for 1-Nearest Neighbors with euclidean distance, 'appr' for approximate k-Nearest Neighbors.
            distance_threshold (float): The threshold above which faces are recognized as being similar.
            method (str): Type of graph to use for the k-nearest neighbor approximation. See https://github.com/nmslib/nmslib/blob/master/manual/methods.md for available options. Only necessary if algorithm = 'appr'.
            space (str): Similarity measure to use in the space. Only necessary if algorithm = 'appr'.
            index_path (str): Path to an existing nmslib-index. Only necessary if algorithm = 'appr'.
            k (int): The number of k-nearest neighbors to consider for the detection. Only necessary if algorithm = 'appr'.
            recognize_by (str): Recognize by 'second' or 'frame'.

        Returns:
            entities (list): Entities found in the video.
        """
        if algorithm == 'appr':
            detector = ApproximateKNearestNeighbors(method,
                                                    space,
                                                    distance_threshold,
                                                    index_path,
                                                    k)
        elif algorithm == '1nn':
            detector = None
        else:
            raise Exception('Unknown Predictor')

        self.path_to_video = download_youtube_video(self.url, tempfile.gettempdir())
        return self.face_detection.recognize_video(self.path_to_video, detector, distance_threshold, recognize_by)

    
