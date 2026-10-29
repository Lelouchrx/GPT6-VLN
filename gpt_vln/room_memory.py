


class RoomStageLedger:
    def __init__(self):
        self.records={}

    def update(self,node_id,room,relation,stage,evidence_frame,allowed_frames):
        if relation=='none':return None
        if evidence_frame not in set(allowed_frames):
            raise ValueError('Room-stage hypothesis must cite a supplied panorama')
        if relation=='confirmed' and (not room.strip() or type(stage) is not int):
            raise ValueError('Confirmed room-stage claim needs a room and stage')
        row={'node_id':node_id,'room_hypothesis':room[:80],
             'instruction_relation':relation,'stage_hypothesis':stage,
             'evidence_frame':evidence_frame,
             'authority':'Claude visual hypothesis, not Matterport ground truth'}
        self.records[node_id]=row
        return row

    def context(self):
        return [self.records[key] for key in sorted(self.records)]
