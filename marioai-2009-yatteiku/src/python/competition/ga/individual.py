from enum import IntEnum

import numpy

class ActionEnum(IntEnum):
    right = 0
    jump = 1
    speed = 2


class LevelSceneCode(IntEnum):
    nothing = 0
    coin = 2
    block = -60
    princess = 5
    ground = 1


class LevelScene(object):
    @classmethod
    def is_obstacle(cls, code):
        return not (code == LevelSceneCode.nothing
                    or code == LevelSceneCode.coin
                    or code == LevelSceneCode.princess)


class Individual(object):
    n_actions = 5
    length = 2**9
    gene_size = 2**n_actions

    def __init__(self, data=None, random=False):
        self.data = data if data is not None else numpy.random.randint(Individual.gene_size, size=self.length) if random else numpy.zeros(self.length, int)

    def to_list(self):
        return list(self.data)

    @classmethod
    def from_list(cls, lst):
        return Individual(data=numpy.array(lst))

    def gene_index_from_levelscene(self, levelscene, isMarioOnGround, mayMarioJump):
        near_cells = [
            levelscene[10][10], # 左上
            levelscene[10][11], # 上
            levelscene[10][12], # 右上
            levelscene[11][10], # 左
            levelscene[11][12], # 右
            levelscene[12][10], # 左下
            levelscene[12][12], # 右下
        ]
        cells_info = list(map(lambda cell: '1' if LevelScene.is_obstacle(cell) else '0', near_cells))
        mario_info = [str(int(isMarioOnGround)), str(int(mayMarioJump))]
        return int(''.join(cells_info + mario_info), 2)

    def action(self, levelscene, isMarioOnGround, mayMarioJump):
        fmt = '{0:0' + str(Individual.gene_size.bit_length()) + 'b}'
        digits = fmt.format(self.data[self.gene_index_from_levelscene(levelscene, isMarioOnGround, mayMarioJump)])
        
        # # 右側に障害物があるとき
        # near_obstacle_ahead = LevelScene.is_obstacle(levelscene[11][12])
        # right_offset = 2
        # far_obstacle_ahead = any(LevelScene.is_obstacle(levelscene[11][x]) for x in range(12, 12 + right_offset))
        # if near_obstacle_ahead or far_obstacle_ahead:
        #     digits_l = list(digits)
        #     digits_l[ActionEnum.jump] = '1'
            
        #     # 右側に障害物があるとき
        #     if near_obstacle_ahead:
        #         digits_l[ActionEnum.right] = '0'
        #         digits_l[ActionEnum.speed] = '0'
        #         digits = ''.join(digits_l)
        
        #     # 右側の少し先に障害物があるとき
        #     else:
        #         digits_l[ActionEnum.right] = '1'
        #         digits_l[ActionEnum.speed] = '1'
        #         digits = ''.join(digits_l)
            
        return list(map(lambda d: int(d), digits))
