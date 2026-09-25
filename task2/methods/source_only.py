from methods.base import AdaptationMethod


class SourceOnlyMethod(AdaptationMethod):
    """Plain cross-entropy over the three labeled source domains. Ignores the
    target batch entirely (it is still fed the same domain-balanced source
    batches as every adaptation method, per 'keep ... source sampling ...
    fixed across methods')."""

    def compute_step(self, source_batches: dict, target_batch, progress: float) -> dict:
        loss, feat, logits, y = self._source_classification_loss(source_batches)
        return {"loss": loss, "cls_loss": loss.item()}
