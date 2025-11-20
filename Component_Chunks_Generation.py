import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import lru_cache

# Настройка логирования
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Класс для представления чанка"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    node_ids: List[str]
    chunk_type: str
    token_count: int

@dataclass
class DocumentNode:
    """Узел документа из DocumentGraph"""
    node_id: str
    content: str
    node_type: str  # 'paragraph', 'heading', 'table', 'image_caption', etc.
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None

class ChunkGenerator:
    """
    Компонент генерации чанков для CompactRAG
    Оптимизирован для скорости и точности разбиения документов
    """
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 200, overlap_size: int = 100, use_semantic_splitting: bool = True, max_workers: int = 4):
        """
        Args:
            max_chunk_size: Максимальный размер чанка в символах
            min_chunk_size: Минимальный размер чанка в символах
            overlap_size: Размер перекрытия между чанками
            use_semantic_splitting: Использовать семантическое разбиение
            max_workers: Количество потоков для параллельной обработки
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.use_semantic_splitting = use_semantic_splitting
        self.max_workers = max_workers
        
        # Кэш для часто используемых вычислений
        self._sentence_cache = {}
        
        # Регулярные выражения для семантического разбиения
        header_pattern = r'^#+\s+.+$'
        self.sentence_endings = re.compile(r'[.!?।॥。！？]+\s*')
        self.semantic_boundaries = re.compile(
        r'[\n\r]+|' + header_pattern + r'|\d+\.\s+',
        re.MULTILINE  # Важно: ^ и $ работают на каждую строку
        )
    
    def generate_chunks_from_document_graph(self, document_nodes: List[DocumentNode]) -> List[Chunk]:
        """
        Основной метод генерации чанков из DocumentGraph
        
        Args:
            document_nodes: Список узлов документа из DocumentGraph
            
        Returns:
            List[Chunk]: Список сгенерированных чанков
        """
        logger.info(f"Начало генерации чанков из {len(document_nodes)} узлов")
        
        # Группируем узлы по семантическим блокам
        semantic_blocks = self._group_nodes_into_semantic_blocks(document_nodes)
        
        # Генерируем чанки из семантических блоков
        chunks = self._generate_chunks_from_blocks(semantic_blocks)
        
        logger.info(f"Сгенерировано {len(chunks)} чанков")
        return chunks
    
    def _group_nodes_into_semantic_blocks(self, nodes: List[DocumentNode]) -> List[List[DocumentNode]]:
        """
        Группирует узлы в семантические блоки для сохранения контекста
        """
        blocks = []
        current_block = []
        current_size = 0
        
        for node in nodes:
            node_size = len(node.content)
            
            # Определяем, является ли узел границей семантического блока
            is_boundary = self._is_semantic_boundary(node)
            
            if current_block and (current_size + node_size > self.max_chunk_size or is_boundary):
                # Сохраняем текущий блок и начинаем новый
                if current_size >= self.min_chunk_size:
                    blocks.append(current_block.copy())
                else:
                    # Объединяем с предыдущим блоком если текущий слишком мал
                    if blocks:
                        blocks[-1].extend(current_block)
                    else:
                        blocks.append(current_block.copy())
                
                current_block = []
                current_size = 0
            
            current_block.append(node)
            current_size += node_size
        
        # Добавляем последний блок
        if current_block:
            if current_size >= self.min_chunk_size or not blocks:
                blocks.append(current_block)
            else:
                blocks[-1].extend(current_block)
        
        return blocks
    
    def _is_semantic_boundary(self, node: DocumentNode) -> bool:
        """
        Определяет, является ли узел границей семантического блока
        """
        # Заголовки всегда являются границами
        if node.node_type in ['heading', 'title', 'subtitle']:
            return True
        
        # Таблицы и изображения - границы
        if node.node_type in ['table', 'image', 'figure']:
            return True
        
        # Длинные паузы в тексте (много переносов строк)
        if node.node_type == 'paragraph' and '\n\n' in node.content:
            return True
        
        return False
    
    def _generate_chunks_from_blocks(self, blocks: List[List[DocumentNode]]) -> List[Chunk]:
        """
        Генерирует чанки из семантических блоков
        """
        all_chunks = []
        
        # Обрабатываем блоки параллельно для увеличения скорости
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_block = {
                executor.submit(self._process_single_block, block, i): (block, i)
                for i, block in enumerate(blocks)
            }
            
            for future in as_completed(future_to_block):
                block, block_idx = future_to_block[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Ошибка при обработке блока {block_idx}: {e}")
                    # Резервная обработка при ошибке
                    backup_chunks = self._process_block_fallback(block, block_idx)
                    all_chunks.extend(backup_chunks)
        
        # Сортируем чанки по порядку в документе
        all_chunks.sort(key = lambda x: x.metadata.get('document_order', 0))
        return all_chunks
    
    def _process_single_block(self, block: List[DocumentNode], block_index: int) -> List[Chunk]:
        """
        Обрабатывает один семантический блок и генерирует чанки
        """
        if not block:
            return []
        
        # Объединяем содержимое блока
        block_content = self._merge_block_content(block)
        block_node_ids = [node.node_id for node in block]
        
        # Если блок меньше максимального размера - создаем один чанк
        if len(block_content) <= self.max_chunk_size:
            return [self._create_chunk(
                content = block_content,
                node_ids = block_node_ids,
                chunk_type = self._determine_chunk_type(block),
                block_index = block_index,
                chunk_index = 0
            )]
        
        # Разбиваем большой блок на чанки
        return self._split_large_block(
            block_content, block_node_ids, block, block_index
        )
    
    def _merge_block_content(self, block: List[DocumentNode]) -> str:
        """
        Объединяет содержимое узлов блока с учетом их типов
        """
        content_parts = []
        
        for node in block:
            if node.node_type == 'heading':
                content_parts.append(f"\n# {node.content}\n")
            elif node.node_type == 'table':
                content_parts.append(f"\n[TABLE]\n{node.content}\n[/TABLE]\n")
            elif node.node_type == 'image_caption':
                content_parts.append(f"\n[IMAGE: {node.content}]\n")
            else:
                content_parts.append(node.content)
        
        return '\n'.join(content_parts)
    
    def _split_large_block(self, content: str, node_ids: List[str], original_block: List[DocumentNode], block_index: int) -> List[Chunk]:
        """
        Разбивает большой блок на несколько чанков с перекрытием
        """
        chunks = []
        
        if self.use_semantic_splitting:
            # Семантическое разбиение по предложениям и границам
            split_points = self._find_semantic_split_points(content)
        else:
            # Простое разбиение по фиксированному размеру
            split_points = self._find_fixed_split_points(content)
        
        start_pos = 0
        chunk_index = 0
        
        while start_pos < len(content):
            # Определяем конечную позицию для текущего чанка
            if chunk_index < len(split_points) - 1:
                end_pos = split_points[chunk_index + 1]
            else:
                end_pos = len(content)
            
            # Извлекаем содержимое чанка
            chunk_content = content[start_pos:end_pos].strip()
            
            if len(chunk_content) >= self.min_chunk_size:
                # Определяем какие узлы попали в чанк
                chunk_node_ids = self._find_nodes_in_chunk(
                    original_block, start_pos, end_pos, content
                )
                
                chunk = self._create_chunk(
                    content = chunk_content,
                    node_ids = chunk_node_ids,
                    chunk_type = self._determine_chunk_type(original_block),
                    block_index = block_index,
                    chunk_index = chunk_index
                )
                chunks.append(chunk)
            
            # Переходим к следующему чанку с перекрытием
            if chunk_index < len(split_points) - 1:
                next_start = split_points[chunk_index + 1] - self.overlap_size
                start_pos = max(start_pos + self.max_chunk_size - self.overlap_size, next_start)
            else:
                break
            
            chunk_index += 1
        
        return chunks
    
    @lru_cache(maxsize = 1000)
    def _find_semantic_split_points(self, content: str) -> List[int]:
        """
        Находит семантически осмысленные точки разбиения текста
        Использует кэширование для повторяющихся текстов
        """
        split_points = [0]
        
        # Ищем концы предложений
        sentence_matches = list(self.sentence_endings.finditer(content))
        
        current_pos = 0
        for match in sentence_matches:
            pos = match.end()
            chunk_size = pos - current_pos
            
            # Если накопилось достаточно текста для чанка
            if chunk_size >= self.max_chunk_size * 0.7:  # 70% от максимального размера
                split_points.append(pos)
                current_pos = pos
        
        # Всегда добавляем конец контента
        if split_points[-1] != len(content):
            split_points.append(len(content))
        
        return split_points
    
    def _find_fixed_split_points(self, content: str) -> List[int]:
        """
        Находит точки разбиения по фиксированному размеру
        """
        split_points = []
        content_length = len(content)
        
        for i in range(0, content_length, self.max_chunk_size - self.overlap_size):
            split_points.append(i)
        
        if split_points[-1] != content_length:
            split_points.append(content_length)
        
        return split_points
    
    def _find_nodes_in_chunk(self, nodes: List[DocumentNode], start_pos: int, end_pos: int, full_content: str) -> List[str]:
        """
        Определяет какие узлы документа попадают в чанк
        """
        chunk_node_ids = []
        current_pos = 0
        
        for node in nodes:
            node_content = self._get_node_content_for_matching(node)
            node_length = len(node_content)
            
            # Проверяем пересекается ли узел с чанком
            node_start = full_content.find(node_content, current_pos)
            if node_start == -1:
                continue
                
            node_end = node_start + node_length
            
            if (node_start < end_pos and node_end > start_pos):
                chunk_node_ids.append(node.node_id)
            
            current_pos = node_end
        
        return chunk_node_ids
    
    def _get_node_content_for_matching(self, node: DocumentNode) -> str:
        """
        Возвращает содержимое узла для сопоставления при разбиении
        """
        if node.node_type == 'heading':
            return f"# {node.content}"
        elif node.node_type == 'table':
            return f"[TABLE]\n{node.content}\n[/TABLE]"
        elif node.node_type == 'image_caption':
            return f"[IMAGE: {node.content}]"
        else:
            return node.content
    
    def _determine_chunk_type(self, nodes: List[DocumentNode]) -> str:
        """
        Определяет тип чанка на основе содержащихся узлов
        """
        node_types = [node.node_type for node in nodes]
        
        if 'table' in node_types:
            return 'table_chunk'
        elif 'image' in node_types or 'image_caption' in node_types:
            return 'visual_chunk'
        elif 'heading' in node_types:
            return 'heading_chunk'
        else:
            return 'text_chunk'
    
    def _create_chunk(self, content: str, node_ids: List[str], chunk_type: str, block_index: int, chunk_index: int) -> Chunk:
        """
        Создает объект чанка с уникальным ID и метаданными
        """
        # Генерируем уникальный ID на основе содержимого
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
        chunk_id = f"chunk_{block_index}_{chunk_index}_{content_hash}"
        
        # Оцениваем количество токенов (приблизительно)
        token_count = self._estimate_token_count(content)
        
        metadata = {
            'chunk_type': chunk_type,
            'block_index': block_index,
            'chunk_index': chunk_index,
            'content_length': len(content),
            'node_count': len(node_ids),
            'document_order': block_index * 1000 + chunk_index,
            'created_at': self._get_timestamp()
        }
        
        return Chunk(
            chunk_id = chunk_id,
            content = content,
            metadata = metadata,
            node_ids = node_ids,
            chunk_type = chunk_type,
            token_count = token_count
        )
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Быстрая оценка количества токенов (приблизительно)
        В production можно заменить на точную токенизацию
        """
        # Приблизительная оценка: 1 токен ≈ 4 символа для английского
        # Для русского может быть немного по-другому, но это быстрая оценка
        return len(text) // 4
    
    def _get_timestamp(self) -> str:
        """Возвращает текущую временную метку"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _process_block_fallback(self, block: List[DocumentNode], block_index: int) -> List[Chunk]:
        """
        Резервный метод обработки блока при ошибках в основном алгоритме
        """
        logger.warning(f"Использование резервного метода для блока {block_index}")
        
        try:
            content = self._merge_block_content(block)
            node_ids = [node.node_id for node in block]
            
            # Простое разбиение на равные части
            chunks = []
            for i in range(0, len(content), self.max_chunk_size - self.overlap_size):
                chunk_content = content[i:i + self.max_chunk_size]
                if len(chunk_content) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        content = chunk_content,
                        node_ids= node_ids,
                        chunk_type = 'fallback_chunk',
                        block_index = block_index,
                        chunk_index = i // self.max_chunk_size
                    )
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Ошибка в резервном методе: {e}")
            return []


# Утилиты для работы с чанками
class ChunkUtils:
    """Вспомогательные утилиты для работы с чанками"""
    
    @staticmethod
    def filter_chunks_by_type(chunks: List[Chunk], chunk_type: str) -> List[Chunk]:
        """Фильтрует чанки по типу"""
        return [chunk for chunk in chunks if chunk.chunk_type == chunk_type]
    
    @staticmethod
    def get_chunks_by_node(chunks: List[Chunk], node_id: str) -> List[Chunk]:
        """Находит все чанки, содержащие указанный узел"""
        return [chunk for chunk in chunks if node_id in chunk.node_ids]
    
    @staticmethod
    def calculate_chunks_statistics(chunks: List[Chunk]) -> Dict[str, Any]:
        """Рассчитывает статистику по чанкам"""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_tokens = sum(chunk.token_count for chunk in chunks)
        total_content = sum(len(chunk.content) for chunk in chunks)
        
        type_distribution = {}
        for chunk in chunks:
            type_distribution[chunk.chunk_type] = type_distribution.get(chunk.chunk_type, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'total_tokens': total_tokens,
            'total_content_length': total_content,
            'average_tokens_per_chunk': total_tokens / total_chunks,
            'average_content_length': total_content / total_chunks,
            'type_distribution': type_distribution
        }


# Пример использования
def main():
    """Пример использования компонента генерации чанков"""
    
    # Создаем тестовые узлы документа
    test_nodes = [
        DocumentNode(
            node_id="node_1",
            content="Это заголовок важного раздела",
            node_type="heading",
            metadata={"level": 1}
        ),
        DocumentNode(
            node_id="node_2", 
            content="Это первый абзац текста. Он содержит важную информацию о системе CompactRAG.",
            node_type="paragraph",
            metadata={}
        ),
        DocumentNode(
            node_id="node_3",
            content="А это второй абзац. Он продолжает тему и добавляет дополнительные детали.",
            node_type="paragraph", 
            metadata={}
        ),
        DocumentNode(
            node_id="node_4",
            content="| Параметр | Значение |\n|----------|----------|\n| Скорость | Высокая |\n| Точность | Средняя |",
            node_type="table",
            metadata={}
        )
    ]
    
    # Инициализируем генератор чанков
    chunk_generator = ChunkGenerator(
        max_chunk_size=500,
        min_chunk_size=100,
        overlap_size=50,
        use_semantic_splitting=True,
        max_workers=2
    )
    
    # Генерируем чанки
    chunks = chunk_generator.generate_chunks_from_document_graph(test_nodes)
    
    # Выводим результаты
    print(f"Сгенерировано чанков: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nЧанк {i+1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Тип: {chunk.chunk_type}")
        print(f"  Размер: {len(chunk.content)} символов")
        print(f"  Токены: {chunk.token_count}")
        print(f"  Узлы: {chunk.node_ids}")
        print(f"  Содержимое: {chunk.content[:100]}...")
    
    # Статистика
    stats = ChunkUtils.calculate_chunks_statistics(chunks)
    print(f"\nСтатистика: {stats}")

if __name__ == "__main__":
    main()